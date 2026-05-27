package service

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type fakeTOSImageClient struct {
	putInputs          []tosPutObjectInput
	preSignedGetBucket string
	preSignedGetKey    string
	preSignedGetExpiry int64
	putErr             error
	signedURL          string
}

func (f *fakeTOSImageClient) PutObject(ctx context.Context, input tosPutObjectInput) error {
	_ = ctx
	if f.putErr != nil {
		return f.putErr
	}
	copied := input
	copied.Body = append([]byte(nil), input.Body...)
	f.putInputs = append(f.putInputs, copied)
	return nil
}

func (f *fakeTOSImageClient) PreSignedGetURL(ctx context.Context, bucket, key string, expiresSeconds int64) (string, error) {
	_ = ctx
	f.preSignedGetBucket = bucket
	f.preSignedGetKey = key
	f.preSignedGetExpiry = expiresSeconds
	if f.signedURL != "" {
		return f.signedURL, nil
	}
	return "https://signed.example.com/" + bucket + "/" + key, nil
}

func TestTOSImageStorageUploadImageUsesOpenAPIBucketAndPublicURL(t *testing.T) {
	client := &fakeTOSImageClient{}
	storage := newTOSImageStorage(config.GatewayImageTOSConfig{
		Enabled:         true,
		Endpoint:        "tos-cn-beijing.volces.com",
		Region:          "cn-beijing",
		AccessKeyID:     "ak",
		SecretAccessKey: "sk",
		Bucket:          "open-api",
		PublicBaseURL:   "https://cdn.example.com/assets/",
		Prefix:          "/image2/",
	}, client)

	result, err := storage.UploadImage(context.Background(), TOSImageUploadInput{
		Data:        []byte("hello"),
		ContentType: "image/png",
		Extension:   "png",
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "open-api", result.Bucket)
	require.Equal(t, "image/png", result.ContentType)
	require.Len(t, client.putInputs, 1)
	require.Equal(t, "open-api", client.putInputs[0].Bucket)
	require.Equal(t, []byte("hello"), client.putInputs[0].Body)
	require.Equal(t, "image/png", client.putInputs[0].ContentType)
	require.True(t, strings.HasPrefix(client.putInputs[0].Key, "image2/"))
	require.True(t, strings.HasSuffix(client.putInputs[0].Key, ".png"))
	require.Equal(t, "https://cdn.example.com/assets/"+escapeTOSObjectKey(client.putInputs[0].Key), result.URL)
}

func TestTOSImageStorageDefaultsToOpenAPIBucket(t *testing.T) {
	client := &fakeTOSImageClient{}
	storage := newTOSImageStorage(config.GatewayImageTOSConfig{
		Enabled:         true,
		Endpoint:        "tos-cn-beijing.volces.com",
		Region:          "cn-beijing",
		AccessKeyID:     "ak",
		SecretAccessKey: "sk",
		PublicBaseURL:   "https://cdn.example.com",
	}, client)

	result, err := storage.UploadImage(context.Background(), TOSImageUploadInput{
		Data:        []byte("hello"),
		ContentType: "image/png",
		Extension:   "png",
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "open-api", result.Bucket)
	require.Equal(t, "open-api", client.putInputs[0].Bucket)
}

func TestTOSImageStorageUploadImageUsesConfiguredReadLinkExpiry(t *testing.T) {
	client := &fakeTOSImageClient{signedURL: "https://signed.example.com/read"}
	storage := newTOSImageStorage(config.GatewayImageTOSConfig{
		Enabled:                true,
		Endpoint:               "tos-cn-beijing.volces.com",
		Region:                 "cn-beijing",
		AccessKeyID:            "ak",
		SecretAccessKey:        "sk",
		Bucket:                 "open-api",
		Prefix:                 "image2",
		ReadLinkExpiresSeconds: 7200,
	}, client)

	result, err := storage.UploadImage(context.Background(), TOSImageUploadInput{
		Data:        []byte("hello"),
		ContentType: "image/png",
		Extension:   "png",
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://signed.example.com/read", result.URL)
	require.Equal(t, "open-api", client.preSignedGetBucket)
	require.Equal(t, client.putInputs[0].Key, client.preSignedGetKey)
	require.EqualValues(t, 7200, client.preSignedGetExpiry)
}

func TestRewriteOpenAIImagesResponseWithTOSUploadsBase64AsURLWhenRequested(t *testing.T) {
	storage, client := newTestTOSStorage()
	body := []byte(`{"created":1710000007,"data":[{"b64_json":"aGVsbG8=","revised_prompt":"draw a cat","output_format":"png"}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "url")
	require.NoError(t, err)
	require.Equal(t, "https://cdn.example.com/"+escapeTOSObjectKey(client.putInputs[0].Key), gjson.GetBytes(rewritten, "data.0.url").String())
	require.False(t, gjson.GetBytes(rewritten, "data.0.b64_json").Exists())
	require.Equal(t, "draw a cat", gjson.GetBytes(rewritten, "data.0.revised_prompt").String())
	require.Equal(t, []byte("hello"), client.putInputs[0].Body)
}

func TestRewriteOpenAIImagesResponseWithTOSPreservesBase64WhenRequested(t *testing.T) {
	storage, client := newTestTOSStorage()
	body := []byte(`{"created":1710000007,"data":[{"b64_json":"aGVsbG8=","revised_prompt":"draw a cat","output_format":"png"}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "b64_json")
	require.NoError(t, err)
	require.Equal(t, "aGVsbG8=", gjson.GetBytes(rewritten, "data.0.b64_json").String())
	require.False(t, gjson.GetBytes(rewritten, "data.0.url").Exists())
	require.Equal(t, "draw a cat", gjson.GetBytes(rewritten, "data.0.revised_prompt").String())
	require.Equal(t, []byte("hello"), client.putInputs[0].Body)
}

func TestRewriteOpenAIImagesResponseWithTOSUploadsDataURL(t *testing.T) {
	storage, client := newTestTOSStorage()
	body := []byte(`{"created":1710000007,"data":[{"url":"data:image/png;base64,aGVsbG8="}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "url")
	require.NoError(t, err)
	require.Equal(t, "https://cdn.example.com/"+escapeTOSObjectKey(client.putInputs[0].Key), gjson.GetBytes(rewritten, "data.0.url").String())
	require.Equal(t, []byte("hello"), client.putInputs[0].Body)
}

func TestRewriteOpenAIImagesResponseWithTOSLeavesOrdinaryURL(t *testing.T) {
	storage, client := newTestTOSStorage()
	body := []byte(`{"created":1710000007,"data":[{"url":"https://already.example.com/a.png"}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "url")
	require.NoError(t, err)
	require.JSONEq(t, string(body), string(rewritten))
	require.Empty(t, client.putInputs)
}

func TestRewriteOpenAIImagesResponseWithTOSDisabledLeavesBody(t *testing.T) {
	body := []byte(`{"created":1710000007,"data":[{"b64_json":"aGVsbG8="}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, nil, "url")
	require.NoError(t, err)
	require.Equal(t, body, rewritten)
}

func TestRewriteOpenAIImagesResponseWithTOSErrorFails(t *testing.T) {
	storage, _ := newTestTOSStorage()
	storage.client.(*fakeTOSImageClient).putErr = errors.New("boom")
	body := []byte(`{"created":1710000007,"data":[{"b64_json":"aGVsbG8="}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "url")
	require.Error(t, err)
	require.Nil(t, rewritten)
}

func TestRewriteOpenAIImagesResponseWithTOSInvalidBase64Fails(t *testing.T) {
	storage, _ := newTestTOSStorage()
	body := []byte(`{"created":1710000007,"data":[{"b64_json":"not base64!!!"}]}`)

	rewritten, err := rewriteOpenAIImagesResponseWithStorage(context.Background(), body, storage, "url")
	require.Error(t, err)
	require.Nil(t, rewritten)
	require.Contains(t, err.Error(), "decode image payload")
}

func TestRewriteOpenAIImagesCompletedEventWithTOS(t *testing.T) {
	storage, client := newTestTOSStorage()
	payload := []byte(`{"type":"image_generation.completed","created_at":1710000001,"b64_json":"ZmluYWw=","url":"data:image/png;base64,ZmluYWw=","output_format":"png"}`)

	rewritten, err := rewriteOpenAIImagesEventPayloadWithStorage(context.Background(), payload, storage, "url")
	require.NoError(t, err)
	require.Equal(t, "https://cdn.example.com/"+escapeTOSObjectKey(client.putInputs[0].Key), gjson.GetBytes(rewritten, "url").String())
	require.False(t, gjson.GetBytes(rewritten, "b64_json").Exists())
	require.Equal(t, []byte("final"), client.putInputs[0].Body)
}

func TestRewriteOpenAIImagesCompletedEventWithTOSPreservesBase64WhenRequested(t *testing.T) {
	storage, client := newTestTOSStorage()
	payload := []byte(`{"type":"image_generation.completed","created_at":1710000001,"b64_json":"ZmluYWw=","url":"data:image/png;base64,ZmluYWw=","output_format":"png"}`)

	rewritten, err := rewriteOpenAIImagesEventPayloadWithStorage(context.Background(), payload, storage, "b64_json")
	require.NoError(t, err)
	require.Equal(t, "ZmluYWw=", gjson.GetBytes(rewritten, "b64_json").String())
	require.False(t, gjson.GetBytes(rewritten, "url").Exists())
	require.Equal(t, []byte("final"), client.putInputs[0].Body)
}

func TestRewriteOpenAIImagesMultilineSSEEventWithTOS(t *testing.T) {
	storage, client := newTestTOSStorage()
	lines := []string{
		"event: image_generation.completed",
		`data: {"type":"image_generation.completed",`,
		`data: "created_at":1710000001,`,
		`data: "b64_json":"ZmluYWw=",`,
		`data: "output_format":"png"}`,
		"",
	}

	rewritten, err := rewriteOpenAIImagesSSEEventLinesWithStorage(context.Background(), lines, storage, "url")
	require.NoError(t, err)
	require.Len(t, rewritten, 3)
	require.Equal(t, "event: image_generation.completed", rewritten[0])
	data := strings.TrimPrefix(rewritten[1], "data: ")
	require.Equal(t, "https://cdn.example.com/"+escapeTOSObjectKey(client.putInputs[0].Key), gjson.Get(data, "url").String())
	require.False(t, gjson.Get(data, "b64_json").Exists())
	require.Equal(t, "", rewritten[2])
	require.Equal(t, []byte("final"), client.putInputs[0].Body)
}

func TestTOSImageStorageBuildsVolcenginePublicURLWhenNoBaseConfigured(t *testing.T) {
	client := &fakeTOSImageClient{}
	storage := newTOSImageStorage(config.GatewayImageTOSConfig{
		Enabled:         true,
		Endpoint:        "tos-cn-beijing.volces.com",
		Region:          "cn-beijing",
		AccessKeyID:     "ak",
		SecretAccessKey: "sk",
		Bucket:          "open-api",
		Prefix:          "image2",
	}, client)

	result, err := storage.UploadImage(context.Background(), TOSImageUploadInput{
		Data:        []byte("hello"),
		ContentType: "image/png",
		Extension:   "png",
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "https://open-api.tos-cn-beijing.volces.com/"+escapeTOSObjectKey(client.putInputs[0].Key), result.URL)
	require.Empty(t, client.preSignedGetKey)
}

func newTestTOSStorage() (*TOSImageStorage, *fakeTOSImageClient) {
	client := &fakeTOSImageClient{}
	storage := newTOSImageStorage(config.GatewayImageTOSConfig{
		Enabled:         true,
		Endpoint:        "tos-cn-beijing.volces.com",
		Region:          "cn-beijing",
		AccessKeyID:     "ak",
		SecretAccessKey: "sk",
		Bucket:          "open-api",
		PublicBaseURL:   "https://cdn.example.com",
		Prefix:          "image2",
	}, client)
	return storage, client
}
