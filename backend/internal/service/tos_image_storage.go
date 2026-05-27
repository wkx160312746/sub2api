package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/volcengine/ve-tos-golang-sdk/v2/tos"
	"github.com/volcengine/ve-tos-golang-sdk/v2/tos/enum"
)

type openAIImageStorage interface {
	Enabled() bool
	UploadImage(ctx context.Context, input TOSImageUploadInput) (*TOSImageUploadResult, error)
}

type tosImageClient interface {
	PutObject(ctx context.Context, input tosPutObjectInput) error
	PreSignedGetURL(ctx context.Context, bucket, key string, expiresSeconds int64) (string, error)
}

type tosPutObjectInput struct {
	Bucket        string
	Key           string
	Body          []byte
	ContentType   string
	ContentLength int64
}

type TOSImageUploadInput struct {
	Data        []byte
	ContentType string
	Extension   string
}

type TOSImageUploadResult struct {
	URL         string
	Bucket      string
	Key         string
	ContentType string
	ByteSize    int64
}

type TOSImageStorage struct {
	cfg      config.GatewayImageTOSConfig
	client   tosImageClient
	clientMu sync.Mutex
}

func newTOSImageStorage(cfg config.GatewayImageTOSConfig, client tosImageClient) *TOSImageStorage {
	cfg.Endpoint = strings.TrimSpace(cfg.Endpoint)
	cfg.Region = strings.TrimSpace(cfg.Region)
	cfg.AccessKeyID = strings.TrimSpace(cfg.AccessKeyID)
	cfg.SecretAccessKey = strings.TrimSpace(cfg.SecretAccessKey)
	cfg.Bucket = strings.TrimSpace(cfg.Bucket)
	if cfg.Bucket == "" {
		cfg.Bucket = "open-api"
	}
	cfg.PublicBaseURL = strings.TrimRight(strings.TrimSpace(cfg.PublicBaseURL), "/")
	cfg.Prefix = strings.Trim(strings.TrimSpace(cfg.Prefix), "/")
	return &TOSImageStorage{cfg: cfg, client: client}
}

func (s *TOSImageStorage) Enabled() bool {
	if s == nil || !s.cfg.Enabled {
		return false
	}
	return strings.TrimSpace(s.cfg.Endpoint) != "" &&
		strings.TrimSpace(s.cfg.Region) != "" &&
		strings.TrimSpace(s.cfg.AccessKeyID) != "" &&
		strings.TrimSpace(s.cfg.SecretAccessKey) != "" &&
		strings.TrimSpace(s.cfg.Bucket) != ""
}

func (s *TOSImageStorage) UploadImage(ctx context.Context, input TOSImageUploadInput) (*TOSImageUploadResult, error) {
	if !s.Enabled() {
		return nil, errors.New("image TOS storage is not enabled or incomplete")
	}
	if len(input.Data) == 0 {
		return nil, errors.New("image data is empty")
	}
	contentType := strings.TrimSpace(input.ContentType)
	if contentType == "" {
		contentType = http.DetectContentType(input.Data)
	}
	ext := openAIImageTOSExtension(input.Extension, contentType)
	key := s.objectKey(ext)
	client, err := s.imageClient()
	if err != nil {
		return nil, err
	}
	if err := client.PutObject(ctx, tosPutObjectInput{
		Bucket:        s.cfg.Bucket,
		Key:           key,
		Body:          input.Data,
		ContentType:   contentType,
		ContentLength: int64(len(input.Data)),
	}); err != nil {
		return nil, fmt.Errorf("upload image to TOS: %w", err)
	}
	readURL, err := s.readURL(ctx, client, key)
	if err != nil {
		return nil, err
	}
	return &TOSImageUploadResult{
		URL:         readURL,
		Bucket:      s.cfg.Bucket,
		Key:         key,
		ContentType: contentType,
		ByteSize:    int64(len(input.Data)),
	}, nil
}

func (s *TOSImageStorage) imageClient() (tosImageClient, error) {
	if s.client != nil {
		return s.client, nil
	}
	s.clientMu.Lock()
	defer s.clientMu.Unlock()
	if s.client != nil {
		return s.client, nil
	}
	created, err := newVolcengineTOSImageClient(s.cfg)
	if err != nil {
		return nil, err
	}
	s.client = created
	return s.client, nil
}

func (s *TOSImageStorage) objectKey(ext string) string {
	if ext == "" {
		ext = "png"
	}
	name := uuid.NewString() + "." + ext
	datePath := time.Now().UTC().Format("2006/01/02")
	parts := []string{s.cfg.Prefix, datePath, name}
	clean := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.Trim(part, "/"); trimmed != "" {
			clean = append(clean, trimmed)
		}
	}
	return path.Join(clean...)
}

func (s *TOSImageStorage) readURL(ctx context.Context, client tosImageClient, key string) (string, error) {
	if base := strings.TrimRight(strings.TrimSpace(s.cfg.PublicBaseURL), "/"); base != "" {
		return base + "/" + escapeTOSObjectKey(key), nil
	}
	if s.cfg.ReadLinkExpiresSeconds <= 0 {
		return "", errors.New("gateway.image_tos.public_base_url or read_link_expires_seconds is required")
	}
	signed, err := client.PreSignedGetURL(ctx, s.cfg.Bucket, key, int64(s.cfg.ReadLinkExpiresSeconds))
	if err != nil {
		return "", fmt.Errorf("generate TOS read URL: %w", err)
	}
	return signed, nil
}

func openAIImageTOSExtension(values ...string) string {
	for _, value := range values {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if normalized == "" {
			continue
		}
		if mediaType, _, err := mime.ParseMediaType(normalized); err == nil {
			normalized = mediaType
		}
		switch normalized {
		case "image/png", "png":
			return "png"
		case "image/jpeg", "image/jpg", "jpeg", "jpg":
			return "jpg"
		case "image/webp", "webp":
			return "webp"
		}
	}
	return "png"
}

func escapeTOSObjectKey(key string) string {
	parts := strings.Split(key, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

type volcengineTOSImageClient struct {
	client *tos.ClientV2
}

func newVolcengineTOSImageClient(cfg config.GatewayImageTOSConfig) (*volcengineTOSImageClient, error) {
	client, err := tos.NewClientV2(
		strings.TrimSpace(cfg.Endpoint),
		tos.WithRegion(strings.TrimSpace(cfg.Region)),
		tos.WithCredentials(tos.NewStaticCredentials(strings.TrimSpace(cfg.AccessKeyID), strings.TrimSpace(cfg.SecretAccessKey))),
	)
	if err != nil {
		return nil, fmt.Errorf("initialize TOS client: %w", err)
	}
	return &volcengineTOSImageClient{client: client}, nil
}

func (c *volcengineTOSImageClient) PutObject(ctx context.Context, input tosPutObjectInput) error {
	if c == nil || c.client == nil {
		return errors.New("TOS client is nil")
	}
	_, err := c.client.PutObjectV2(ctx, &tos.PutObjectV2Input{
		PutObjectBasicInput: tos.PutObjectBasicInput{
			Bucket:        input.Bucket,
			Key:           input.Key,
			ContentLength: input.ContentLength,
			ContentType:   input.ContentType,
		},
		Content: bytes.NewReader(input.Body),
	})
	return err
}

func (c *volcengineTOSImageClient) PreSignedGetURL(ctx context.Context, bucket, key string, expiresSeconds int64) (string, error) {
	_ = ctx
	if c == nil || c.client == nil {
		return "", errors.New("TOS client is nil")
	}
	return c.client.Client.PreSignedURL(string(enum.HttpMethodGet), bucket, key, time.Duration(expiresSeconds)*time.Second)
}

func (s *OpenAIGatewayService) imageStorage() openAIImageStorage {
	if s == nil || s.cfg == nil {
		return nil
	}
	if s.imageTOSStorage != nil {
		return s.imageTOSStorage
	}
	s.imageTOSStorageOnce.Do(func() {
		storage := newTOSImageStorage(s.cfg.Gateway.ImageTOS, nil)
		if storage.Enabled() {
			s.imageTOSStorage = storage
		}
	})
	return s.imageTOSStorage
}

func (s *OpenAIGatewayService) rewriteOpenAIImagesResponseWithTOS(ctx context.Context, body []byte) ([]byte, error) {
	return rewriteOpenAIImagesResponseWithStorage(ctx, body, s.imageStorage())
}

func (s *OpenAIGatewayService) rewriteOpenAIImagesEventPayloadWithTOS(ctx context.Context, payload []byte) ([]byte, error) {
	return rewriteOpenAIImagesEventPayloadWithStorage(ctx, payload, s.imageStorage())
}

func rewriteOpenAIImagesResponseWithStorage(ctx context.Context, body []byte, storage openAIImageStorage) ([]byte, error) {
	if storage == nil || !storage.Enabled() || len(body) == 0 || !gjson.ValidBytes(body) {
		return body, nil
	}
	out := body
	items := gjson.GetBytes(body, "data")
	if !items.IsArray() {
		return body, nil
	}
	for index, item := range items.Array() {
		rewritten, err := rewriteOpenAIImagesItemWithStorage(ctx, out, fmt.Sprintf("data.%d", index), item, storage)
		if err != nil {
			return nil, err
		}
		out = rewritten
	}
	return out, nil
}

func rewriteOpenAIImagesEventPayloadWithStorage(ctx context.Context, payload []byte, storage openAIImageStorage) ([]byte, error) {
	if storage == nil || !storage.Enabled() || len(payload) == 0 || !gjson.ValidBytes(payload) {
		return payload, nil
	}
	eventType := strings.TrimSpace(gjson.GetBytes(payload, "type").String())
	if eventType == "" || !strings.HasSuffix(eventType, ".completed") {
		return payload, nil
	}
	return rewriteOpenAIImagesItemWithStorage(ctx, payload, "", gjson.ParseBytes(payload), storage)
}

func rewriteOpenAIImagesSSEEventLinesWithStorage(ctx context.Context, lines []string, storage openAIImageStorage) ([]string, error) {
	if storage == nil || !storage.Enabled() || len(lines) == 0 {
		return lines, nil
	}
	dataLines := make([]string, 0, len(lines))
	dataIndexes := make(map[int]struct{})
	for index, line := range lines {
		data, ok := extractOpenAISSEDataLine(line)
		if !ok {
			continue
		}
		dataLines = append(dataLines, data)
		dataIndexes[index] = struct{}{}
	}
	if len(dataLines) == 0 {
		return lines, nil
	}
	payload := strings.Join(dataLines, "\n")
	rewritten, err := rewriteOpenAIImagesEventPayloadWithStorage(ctx, []byte(payload), storage)
	if err != nil {
		return nil, err
	}
	if bytes.Equal(rewritten, []byte(payload)) {
		return lines, nil
	}
	out := make([]string, 0, len(lines))
	inserted := false
	for index, line := range lines {
		if _, ok := dataIndexes[index]; ok {
			if !inserted {
				out = append(out, "data: "+string(rewritten))
				inserted = true
			}
			continue
		}
		out = append(out, line)
	}
	return out, nil
}

func rewriteOpenAIImagesItemWithStorage(ctx context.Context, body []byte, itemPath string, item gjson.Result, storage openAIImageStorage) ([]byte, error) {
	if !item.Exists() {
		return body, nil
	}
	b64, contentType := openAIImagePayloadBase64AndContentType(item)
	if b64 == "" {
		return body, nil
	}
	decoded, err := base64.StdEncoding.DecodeString(normalizeOpenAIImageBase64ForTOS(b64))
	if err != nil {
		return nil, fmt.Errorf("decode image payload for TOS upload: %w", err)
	}
	format := firstNonEmptyString(item.Get("output_format").String(), item.Get("format").String())
	uploadContentType := firstNonEmptyString(contentType, openAIImageOutputMIMEType(format), http.DetectContentType(decoded))
	uploaded, err := storage.UploadImage(ctx, TOSImageUploadInput{
		Data:        decoded,
		ContentType: uploadContentType,
		Extension:   openAIImageTOSExtension(format, uploadContentType),
	})
	if err != nil {
		return nil, err
	}
	urlPath := joinJSONPath(itemPath, "url")
	out, err := sjson.SetBytes(body, urlPath, uploaded.URL)
	if err != nil {
		return nil, err
	}
	b64Path := joinJSONPath(itemPath, "b64_json")
	if gjson.GetBytes(out, b64Path).Exists() {
		out, err = sjson.DeleteBytes(out, b64Path)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func normalizeOpenAIImageBase64ForTOS(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(strings.ToLower(raw), "data:") {
		if idx := strings.Index(raw, ","); idx >= 0 && idx+1 < len(raw) {
			raw = raw[idx+1:]
		}
	}
	raw = strings.TrimSpace(raw)
	raw = strings.TrimRight(raw, "=")
	return raw + strings.Repeat("=", (4-len(raw)%4)%4)
}

func openAIImagePayloadBase64AndContentType(item gjson.Result) (string, string) {
	if b64 := strings.TrimSpace(item.Get("b64_json").String()); b64 != "" {
		return b64, ""
	}
	rawURL := strings.TrimSpace(item.Get("url").String())
	if !strings.HasPrefix(strings.ToLower(rawURL), "data:") {
		return "", ""
	}
	contentType := ""
	if comma := strings.Index(rawURL, ","); comma >= 0 {
		meta := rawURL[:comma]
		payload := rawURL[comma+1:]
		if semi := strings.Index(meta, ";"); semi >= 0 {
			contentType = strings.TrimPrefix(meta[:semi], "data:")
		} else {
			contentType = strings.TrimPrefix(meta, "data:")
		}
		return payload, contentType
	}
	return "", ""
}

func joinJSONPath(parent, child string) string {
	if strings.TrimSpace(parent) == "" {
		return child
	}
	return parent + "." + child
}
