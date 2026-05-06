package service

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

const (
	OpenAIImageTaskStatusQueued     = "queued"
	OpenAIImageTaskStatusInProgress = "in_progress"
	OpenAIImageTaskStatusCompleted  = "completed"
	OpenAIImageTaskStatusFailed     = "failed"
)

const (
	openAIImageTaskWorkerPollInterval       = 5 * time.Second
	openAIImageTaskStaleRunningAfterSeconds = 30 * 60
)

type OpenAIImageTaskRepository interface {
	CreateTask(ctx context.Context, task *OpenAIImageTask) error
	GetTask(ctx context.Context, id string) (*OpenAIImageTask, error)
	ClaimTask(ctx context.Context, id string, staleRunningAfterSeconds int64) (*OpenAIImageTask, error)
	ClaimNextPendingTask(ctx context.Context, staleRunningAfterSeconds int64) (*OpenAIImageTask, error)
	MarkTaskSucceeded(ctx context.Context, id string, accountID *int64, result map[string]any) error
	MarkTaskFailed(ctx context.Context, id, code, message string) error
}

type OpenAIImageTask struct {
	ID           string            `json:"id"`
	Object       string            `json:"object"`
	Status       string            `json:"status"`
	Progress     int               `json:"progress"`
	CreatedAt    int64             `json:"created_at"`
	StartedAt    *int64            `json:"started_at,omitempty"`
	CompletedAt  *int64            `json:"completed_at,omitempty"`
	Model        string            `json:"model"`
	ImageTier    string            `json:"image_tier,omitempty"`
	AspectRatio  string            `json:"aspect_ratio,omitempty"`
	TargetSize   string            `json:"target_size,omitempty"`
	UpstreamSize string            `json:"upstream_size,omitempty"`
	Result       map[string]any    `json:"result"`
	Error        map[string]string `json:"error,omitempty"`

	APIKeyID           int64              `json:"-"`
	UserID             int64              `json:"-"`
	GroupID            *int64             `json:"-"`
	AccountID          *int64             `json:"-"`
	ModelAlias         string             `json:"-"`
	TargetWidth        int                `json:"-"`
	TargetHeight       int                `json:"-"`
	OutputFormat       string             `json:"-"`
	ContentType        string             `json:"-"`
	RequestBody        []byte             `json:"-"`
	ChannelMappedModel string             `json:"-"`
	ChannelUsageFields ChannelUsageFields `json:"-"`
	RequestPayloadHash string             `json:"-"`
	UserAgent          string             `json:"-"`
	IPAddress          string             `json:"-"`
	InboundEndpoint    string             `json:"-"`

	parsed       *OpenAIImagesRequest
	apiKey       *APIKey
	subscription *UserSubscription
}

type OpenAIImageTaskService struct {
	gateway         *OpenAIGatewayService
	apiKeySvc       *APIKeyService
	subscriptionSvc *SubscriptionService
	repo            OpenAIImageTaskRepository
	resultDir       string
	mu              sync.RWMutex
	tasks           map[string]*OpenAIImageTask
	queue           chan string
	workers         int
	startOnce       sync.Once
	stopOnce        sync.Once
	stopCh          chan struct{}
}

func NewOpenAIImageTaskService(
	gateway *OpenAIGatewayService,
	apiKeySvc *APIKeyService,
	subscriptionSvc *SubscriptionService,
	repo OpenAIImageTaskRepository,
	cfg *config.Config,
) *OpenAIImageTaskService {
	svc := &OpenAIImageTaskService{
		gateway:         gateway,
		apiKeySvc:       apiKeySvc,
		subscriptionSvc: subscriptionSvc,
		repo:            repo,
		resultDir:       resolveOpenAIImageTaskResultDir(cfg),
		tasks:           make(map[string]*OpenAIImageTask),
		queue:           make(chan string, 128),
		workers:         1,
		stopCh:          make(chan struct{}),
	}
	svc.Start()
	return svc
}

func (s *OpenAIImageTaskService) Start() {
	if s == nil {
		return
	}
	s.startOnce.Do(func() {
		workers := s.workers
		if workers <= 0 {
			workers = 1
		}
		for i := 0; i < workers; i++ {
			go s.worker()
		}
	})
}

func (s *OpenAIImageTaskService) Stop() {
	if s == nil {
		return
	}
	s.stopOnce.Do(func() {
		close(s.stopCh)
	})
}

func (s *OpenAIImageTaskService) CreateTask(
	ctx context.Context,
	parsed *OpenAIImagesRequest,
	body []byte,
	apiKey *APIKey,
	subscription *UserSubscription,
	channelMappedModel string,
	channelUsageFields ChannelUsageFields,
	requestPayloadHash string,
	userAgent string,
	ipAddress string,
	inboundEndpoint string,
) (*OpenAIImageTask, error) {
	if s == nil || s.gateway == nil {
		return nil, fmt.Errorf("image task service is not available")
	}
	if parsed == nil {
		return nil, fmt.Errorf("parsed images request is required")
	}
	if apiKey == nil {
		return nil, fmt.Errorf("api key is required")
	}
	id, err := newOpenAIImageTaskID()
	if err != nil {
		return nil, err
	}
	now := time.Now().Unix()
	contentType := parsed.ContentType
	if contentType == "" {
		contentType = "application/json"
	}
	task := &OpenAIImageTask{
		ID:                 id,
		Object:             "image.task",
		Status:             OpenAIImageTaskStatusQueued,
		Progress:           0,
		CreatedAt:          now,
		Model:              parsed.Model,
		ImageTier:          parsed.ImageTier,
		AspectRatio:        parsed.AspectRatio,
		TargetSize:         openAIImageTaskTargetSize(parsed),
		UpstreamSize:       parsed.UpstreamSize,
		APIKeyID:           apiKey.ID,
		UserID:             apiKey.UserID,
		GroupID:            apiKey.GroupID,
		ModelAlias:         parsed.ModelAlias,
		TargetWidth:        parsed.TargetWidth,
		TargetHeight:       parsed.TargetHeight,
		OutputFormat:       parsed.OutputFormat,
		ContentType:        contentType,
		Result:             nil,
		RequestBody:        append([]byte(nil), body...),
		parsed:             parsed,
		apiKey:             apiKey,
		subscription:       subscription,
		ChannelMappedModel: channelMappedModel,
		ChannelUsageFields: channelUsageFields,
		RequestPayloadHash: requestPayloadHash,
		UserAgent:          userAgent,
		IPAddress:          ipAddress,
		InboundEndpoint:    inboundEndpoint,
	}

	if s.repo != nil {
		if err := s.repo.CreateTask(ctx, task); err != nil {
			return nil, fmt.Errorf("create image task: %w", err)
		}
	}

	s.mu.Lock()
	s.tasks[id] = task
	s.mu.Unlock()

	select {
	case s.queue <- id:
		return task.snapshot(), nil
	default:
		s.markFailed(ctx, id, "queue_full", "image task queue is full")
		return nil, fmt.Errorf("image task queue is full")
	}
}

func (s *OpenAIImageTaskService) GetTask(ctx context.Context, id string) (*OpenAIImageTask, bool, error) {
	if s == nil {
		return nil, false, nil
	}
	if s.repo != nil {
		task, err := s.repo.GetTask(ctx, id)
		if err != nil {
			return nil, false, err
		}
		if task == nil {
			return nil, false, nil
		}
		s.cacheTask(task)
		return task.snapshot(), true, nil
	}
	s.mu.RLock()
	task, ok := s.tasks[id]
	s.mu.RUnlock()
	if !ok || task == nil {
		return nil, false, nil
	}
	return task.snapshot(), true, nil
}

func (s *OpenAIImageTaskService) worker() {
	var ticker *time.Ticker
	var tickerC <-chan time.Time
	if s.repo != nil {
		ticker = time.NewTicker(openAIImageTaskWorkerPollInterval)
		tickerC = ticker.C
		defer ticker.Stop()
	}
	for {
		select {
		case <-s.stopCh:
			return
		case id, ok := <-s.queue:
			if !ok {
				return
			}
			s.runQueuedTask(id)
		case <-tickerC:
			s.claimAndRunNext()
		}
	}
}

func (s *OpenAIImageTaskService) runQueuedTask(id string) {
	ctx := context.Background()
	if s.repo != nil {
		task, err := s.repo.ClaimTask(ctx, id, openAIImageTaskStaleRunningAfterSeconds)
		if err != nil || task == nil {
			return
		}
		s.cacheTask(task)
		s.runClaimedTask(ctx, task)
		return
	}
	task := s.getMutableTask(id)
	if task == nil {
		return
	}
	startedAt := time.Now().Unix()
	s.mu.Lock()
	task.Status = OpenAIImageTaskStatusInProgress
	task.Progress = 10
	task.StartedAt = &startedAt
	s.mu.Unlock()
	s.runClaimedTask(ctx, task)
}

func (s *OpenAIImageTaskService) claimAndRunNext() {
	if s == nil || s.repo == nil {
		return
	}
	ctx := context.Background()
	task, err := s.repo.ClaimNextPendingTask(ctx, openAIImageTaskStaleRunningAfterSeconds)
	if err != nil || task == nil {
		return
	}
	s.cacheTask(task)
	s.runClaimedTask(ctx, task)
}

func (s *OpenAIImageTaskService) runClaimedTask(ctx context.Context, task *OpenAIImageTask) {
	if task == nil {
		return
	}
	if err := s.hydrateTask(ctx, task); err != nil {
		s.markFailed(ctx, task.ID, "task_rehydrate_failed", err.Error())
		return
	}
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = (&http.Request{
		Method: http.MethodPost,
		URL:    &url.URL{Path: openAIImagesGenerationsEndpoint},
		Header: http.Header{},
	}).WithContext(ctx)
	if task.ContentType != "" {
		c.Request.Header.Set("Content-Type", task.ContentType)
	}
	if task.apiKey != nil {
		c.Set("api_key", task.apiKey)
	}
	account, result, err := s.forwardTaskWithFailover(c, task)
	completedAt := time.Now().Unix()
	if err != nil {
		s.markFailed(ctx, task.ID, "upstream_error", err.Error())
		return
	}

	if result != nil && task.apiKey != nil && account != nil {
		_ = s.gateway.RecordUsage(ctx, &OpenAIRecordUsageInput{
			Result:             result,
			APIKey:             task.apiKey,
			User:               task.apiKey.User,
			Account:            account,
			Subscription:       task.subscription,
			InboundEndpoint:    task.InboundEndpoint,
			UpstreamEndpoint:   openAIImageTaskUpstreamEndpoint(task.parsed, account),
			UserAgent:          task.UserAgent,
			IPAddress:          task.IPAddress,
			RequestPayloadHash: task.RequestPayloadHash,
			APIKeyService:      s.apiKeySvc,
			ChannelUsageFields: task.ChannelUsageFields,
		})
	}

	resultPayload := s.openAIImageTaskResultFromRecorder(task, rec)
	var accountID *int64
	if account != nil {
		v := account.ID
		accountID = &v
	}
	s.markSucceeded(ctx, task.ID, accountID, completedAt, resultPayload)
}

func (s *OpenAIImageTaskService) hydrateTask(ctx context.Context, task *OpenAIImageTask) error {
	if task == nil {
		return fmt.Errorf("invalid image task")
	}
	if task.parsed == nil {
		if len(task.RequestBody) == 0 {
			return fmt.Errorf("image task request body is empty")
		}
		contentType := task.ContentType
		if contentType == "" {
			contentType = "application/json"
		}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = (&http.Request{
			Method: http.MethodPost,
			URL:    &url.URL{Path: "/v1/image-tasks"},
			Header: http.Header{"Content-Type": []string{contentType}},
		}).WithContext(ctx)
		parsed, err := s.gateway.ParseOpenAIImagesRequest(c, task.RequestBody)
		if err != nil {
			return err
		}
		task.parsed = parsed
		if task.Model == "" {
			task.Model = parsed.Model
		}
		if task.ImageTier == "" {
			task.ImageTier = parsed.ImageTier
		}
		if task.AspectRatio == "" {
			task.AspectRatio = parsed.AspectRatio
		}
		if task.TargetSize == "" {
			task.TargetSize = openAIImageTaskTargetSize(parsed)
		}
		if task.UpstreamSize == "" {
			task.UpstreamSize = parsed.UpstreamSize
		}
	}
	if task.apiKey == nil {
		if s.apiKeySvc == nil || task.APIKeyID <= 0 {
			return fmt.Errorf("image task api key is unavailable")
		}
		apiKey, err := s.apiKeySvc.GetByID(ctx, task.APIKeyID)
		if err != nil {
			return err
		}
		task.apiKey = apiKey
	}
	if task.subscription == nil && task.apiKey != nil && task.apiKey.Group != nil && task.apiKey.Group.IsSubscriptionType() {
		if s.subscriptionSvc == nil || task.apiKey.GroupID == nil {
			return fmt.Errorf("active subscription is required")
		}
		subscription, err := s.subscriptionSvc.GetActiveSubscription(ctx, task.apiKey.UserID, *task.apiKey.GroupID)
		if err != nil {
			if errors.Is(err, ErrSubscriptionNotFound) {
				return fmt.Errorf("active subscription is required")
			}
			return err
		}
		task.subscription = subscription
	}
	if task.ChannelMappedModel == "" || task.ChannelUsageFields.OriginalModel == "" {
		channelMapping, _ := s.gateway.ResolveChannelMappingAndRestrict(ctx, task.apiKey.GroupID, task.parsed.Model)
		upstreamForUsage := task.parsed.Model
		if channelMapping.MappedModel != "" {
			upstreamForUsage = channelMapping.MappedModel
		}
		task.ChannelMappedModel = channelMapping.MappedModel
		task.ChannelUsageFields = channelMapping.ToUsageFields(task.parsed.Model, upstreamForUsage)
	}
	if task.RequestPayloadHash == "" {
		task.RequestPayloadHash = HashUsageRequestPayload(task.RequestBody)
	}
	if task.InboundEndpoint == "" {
		task.InboundEndpoint = "/v1/image-tasks"
	}
	return nil
}

func (s *OpenAIImageTaskService) forwardTaskWithFailover(c *gin.Context, task *OpenAIImageTask) (*Account, *OpenAIForwardResult, error) {
	if task == nil || task.parsed == nil || task.apiKey == nil {
		return nil, nil, fmt.Errorf("invalid image task")
	}
	var failedAccountIDs map[int64]struct{}
	for attempt := 0; attempt < 3; attempt++ {
		selection, _, err := s.gateway.SelectAccountWithSchedulerForImages(
			context.Background(),
			task.apiKey.GroupID,
			task.parsed.StickySessionSeed(),
			task.parsed.Model,
			failedAccountIDs,
			task.parsed.RequiredCapability,
		)
		if err != nil {
			return nil, nil, err
		}
		if selection == nil || selection.Account == nil {
			return nil, nil, fmt.Errorf("no available compatible accounts")
		}
		account := selection.Account
		result, err := s.gateway.ForwardImages(context.Background(), c, account, task.RequestBody, task.parsed, task.ChannelMappedModel)
		if err == nil {
			s.gateway.ReportOpenAIAccountScheduleResult(account.ID, true, result.FirstTokenMs)
			return account, result, nil
		}
		s.gateway.ReportOpenAIAccountScheduleResult(account.ID, false, nil)
		if failedAccountIDs == nil {
			failedAccountIDs = make(map[int64]struct{})
		}
		failedAccountIDs[account.ID] = struct{}{}
	}
	return nil, nil, fmt.Errorf("image task failed after account failover")
}

func (s *OpenAIImageTaskService) getMutableTask(id string) *OpenAIImageTask {
	s.mu.RLock()
	task := s.tasks[id]
	s.mu.RUnlock()
	return task
}

func (s *OpenAIImageTaskService) cacheTask(task *OpenAIImageTask) {
	if s == nil || task == nil {
		return
	}
	s.mu.Lock()
	s.tasks[task.ID] = task
	s.mu.Unlock()
}

func (s *OpenAIImageTaskService) markSucceeded(ctx context.Context, id string, accountID *int64, completedAt int64, result map[string]any) {
	if s.repo != nil {
		_ = s.repo.MarkTaskSucceeded(ctx, id, accountID, result)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if task := s.tasks[id]; task != nil {
		task.Status = OpenAIImageTaskStatusCompleted
		task.Progress = 100
		task.CompletedAt = &completedAt
		task.AccountID = accountID
		task.Result = result
		task.Error = nil
	}
}

func (s *OpenAIImageTaskService) markFailed(ctx context.Context, id, code, message string) {
	completedAt := time.Now().Unix()
	if s.repo != nil {
		_ = s.repo.MarkTaskFailed(ctx, id, code, message)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if task := s.tasks[id]; task != nil {
		task.Status = OpenAIImageTaskStatusFailed
		task.Progress = 100
		task.CompletedAt = &completedAt
		task.Error = map[string]string{"code": code, "message": message}
	}
}

func (s *OpenAIImageTaskService) openAIImageTaskResultFromRecorder(task *OpenAIImageTask, rec *httptest.ResponseRecorder) map[string]any {
	if rec == nil {
		return nil
	}
	body := rec.Body.String()
	response := any(body)
	var decoded any
	if json.Unmarshal([]byte(body), &decoded) == nil {
		response = decoded
	}
	payload := map[string]any{
		"status_code": rec.Code,
		"mime_type":   rec.Header().Get("Content-Type"),
		"response":    response,
	}
	if asset, err := s.storeOpenAIImageTaskResult(task, []byte(body)); err == nil && asset != nil {
		response = sanitizeOpenAIImageTaskStoredResponse(response, asset)
		payload["url"] = asset.URL
		payload["storage_key"] = asset.StorageKey
		payload["mime_type"] = asset.MimeType
		payload["width"] = asset.Width
		payload["height"] = asset.Height
		if asset.ByteSize > 0 {
			payload["byte_size"] = asset.ByteSize
		}
	}
	return payload
}

func sanitizeOpenAIImageTaskStoredResponse(response any, asset *openAIImageTaskStoredAsset) any {
	body, ok := response.(map[string]any)
	if !ok || asset == nil {
		return response
	}
	data, ok := body["data"].([]any)
	if !ok {
		return response
	}
	for idx, item := range data {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		delete(obj, "b64_json")
		if idx == 0 {
			obj["url"] = asset.URL
		} else if raw, ok := obj["url"].(string); ok && strings.HasPrefix(strings.ToLower(strings.TrimSpace(raw)), "data:image/") {
			delete(obj, "url")
		}
	}
	body["data"] = data
	return body
}

type openAIImageTaskStoredAsset struct {
	URL        string
	StorageKey string
	MimeType   string
	Width      int
	Height     int
	ByteSize   int
}

func (s *OpenAIImageTaskService) storeOpenAIImageTaskResult(task *OpenAIImageTask, body []byte) (*openAIImageTaskStoredAsset, error) {
	if task == nil || len(body) == 0 {
		return nil, nil
	}
	bodyResult := gjson.ParseBytes(body)
	if !bodyResult.Exists() {
		return nil, nil
	}
	item := bodyResult.Get("data.0")
	if !item.Exists() {
		return nil, nil
	}
	b64 := strings.TrimSpace(item.Get("b64_json").String())
	if b64 == "" {
		url := strings.TrimSpace(item.Get("url").String())
		if strings.HasPrefix(strings.ToLower(url), "data:image/") {
			b64 = normalizeOpenAIImageBase64(url)
		}
	}
	if b64 == "" {
		return nil, nil
	}
	decoded, err := base64.StdEncoding.DecodeString(normalizeOpenAIImageBase64(b64))
	if err != nil {
		return nil, err
	}
	ext := openAIImageTaskOutputExtension(task.OutputFormat, item.Get("output_format").String(), "")
	if ext == "" {
		ext = "png"
	}
	stored, mimeType, width, height, err := encodeOpenAIImageTaskTarget(decoded, ext, task.TargetWidth, task.TargetHeight)
	if err != nil {
		stored = decoded
		mimeType = openAIImageTaskMimeType(ext)
	}
	dir := s.resultDir
	if dir == "" {
		dir = resolveOpenAIImageTaskResultDir(nil)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	filename := task.ID + "." + ext
	path := filepath.Join(dir, filename)
	if err := os.WriteFile(path, stored, 0o644); err != nil {
		return nil, err
	}
	if width <= 0 || height <= 0 {
		if cfg, _, err := image.DecodeConfig(bytes.NewReader(stored)); err == nil {
			width = cfg.Width
			height = cfg.Height
		}
	}
	return &openAIImageTaskStoredAsset{
		URL:        "/generated/images/" + filename,
		StorageKey: "generated/images/" + filename,
		MimeType:   mimeType,
		Width:      width,
		Height:     height,
		ByteSize:   len(stored),
	}, nil
}

func (t *OpenAIImageTask) snapshot() *OpenAIImageTask {
	if t == nil {
		return nil
	}
	cp := *t
	cp.RequestBody = nil
	cp.parsed = nil
	cp.apiKey = nil
	cp.subscription = nil
	cp.ChannelUsageFields = ChannelUsageFields{}
	if t.Result != nil {
		cp.Result = make(map[string]any, len(t.Result))
		for k, v := range t.Result {
			cp.Result[k] = v
		}
	}
	if t.Error != nil {
		cp.Error = make(map[string]string, len(t.Error))
		for k, v := range t.Error {
			cp.Error[k] = v
		}
	}
	return &cp
}

func newOpenAIImageTaskID() (string, error) {
	var buf [12]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", fmt.Errorf("generate image task id: %w", err)
	}
	return "imgtask_" + hex.EncodeToString(buf[:]), nil
}

func openAIImageTaskTargetSize(parsed *OpenAIImagesRequest) string {
	if parsed == nil || parsed.TargetWidth <= 0 || parsed.TargetHeight <= 0 {
		return ""
	}
	return fmt.Sprintf("%dx%d", parsed.TargetWidth, parsed.TargetHeight)
}

func resolveOpenAIImageTaskResultDir(cfg *config.Config) string {
	if cfg != nil && strings.TrimSpace(cfg.Gateway.ImageTaskResultDir) != "" {
		return strings.TrimSpace(cfg.Gateway.ImageTaskResultDir)
	}
	if dataDir := strings.TrimSpace(os.Getenv("DATA_DIR")); dataDir != "" {
		return filepath.Join(dataDir, "generated", "images")
	}
	return filepath.Join("/app/data", "generated", "images")
}

func OpenAIImageTaskResultDir(cfg *config.Config) string {
	return resolveOpenAIImageTaskResultDir(cfg)
}

func openAIImageTaskOutputExtension(values ...string) string {
	for _, value := range values {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if strings.Contains(normalized, ";") {
			normalized = strings.TrimSpace(strings.Split(normalized, ";")[0])
		}
		switch normalized {
		case "image/png", "png":
			return "png"
		case "image/jpeg", "image/jpg", "jpeg", "jpg":
			return "jpg"
		case "image/webp", "webp":
			return "png"
		}
	}
	return "png"
}

func openAIImageTaskMimeType(ext string) string {
	switch strings.ToLower(strings.TrimSpace(ext)) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	default:
		return "image/png"
	}
}

func encodeOpenAIImageTaskTarget(decoded []byte, ext string, targetWidth, targetHeight int) ([]byte, string, int, int, error) {
	src, _, err := image.Decode(bytes.NewReader(decoded))
	if err != nil {
		return nil, "", 0, 0, err
	}
	srcBounds := src.Bounds()
	width := srcBounds.Dx()
	height := srcBounds.Dy()
	if targetWidth > 0 && targetHeight > 0 && (width != targetWidth || height != targetHeight) {
		dst := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
		if openAIImageTaskMimeType(ext) == "image/jpeg" {
			draw.Draw(dst, dst.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
			draw.CatmullRom.Scale(dst, dst.Bounds(), src, srcBounds, draw.Over, nil)
		} else {
			draw.CatmullRom.Scale(dst, dst.Bounds(), src, srcBounds, draw.Src, nil)
		}
		src = dst
		width = targetWidth
		height = targetHeight
	}

	var buf bytes.Buffer
	switch openAIImageTaskMimeType(ext) {
	case "image/jpeg":
		if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 92}); err != nil {
			return nil, "", 0, 0, err
		}
		return buf.Bytes(), "image/jpeg", width, height, nil
	default:
		if err := png.Encode(&buf, src); err != nil {
			return nil, "", 0, 0, err
		}
		return buf.Bytes(), "image/png", width, height, nil
	}
}

func openAIImageTaskUpstreamEndpoint(parsed *OpenAIImagesRequest, account *Account) string {
	if parsed == nil {
		return ""
	}
	if account != nil && account.Type == AccountTypeOAuth {
		return chatgptCodexAPIURL
	}
	return parsed.Endpoint
}
