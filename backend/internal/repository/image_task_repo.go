package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

type openAIImageTaskRepository struct {
	sql sqlExecutor
}

func NewOpenAIImageTaskRepository(sqlDB *sql.DB) service.OpenAIImageTaskRepository {
	return &openAIImageTaskRepository{sql: sqlDB}
}

func (r *openAIImageTaskRepository) CreateTask(ctx context.Context, task *service.OpenAIImageTask) error {
	if r == nil || r.sql == nil {
		return nil
	}
	if task == nil {
		return nil
	}
	requestBody := normalizeJSONBBytes(task.RequestBody)
	channelUsageFields, err := json.Marshal(task.ChannelUsageFields)
	if err != nil {
		return fmt.Errorf("marshal channel usage fields: %w", err)
	}
	createdAt := time.Unix(task.CreatedAt, 0)
	query := `
		INSERT INTO image_tasks (
			id, status, progress, request_body, api_key_id, user_id, group_id,
			model, model_alias, image_tier, aspect_ratio, target_width, target_height,
			upstream_size, output_format, content_type, channel_mapped_model,
			channel_usage_fields, request_payload_hash, user_agent, ip_address,
			inbound_endpoint, created_at, updated_at
		) VALUES (
			$1, $2, $3, $4::jsonb, $5, $6, $7,
			$8, $9, $10, $11, $12, $13,
			$14, $15, $16, $17,
			$18::jsonb, $19, $20, $21,
			$22, $23, $23
		)
	`
	_, err = r.sql.ExecContext(
		ctx,
		query,
		task.ID,
		task.Status,
		task.Progress,
		string(requestBody),
		nullInt64Arg(task.APIKeyID),
		nullInt64Arg(task.UserID),
		task.GroupID,
		task.Model,
		task.ModelAlias,
		task.ImageTier,
		task.AspectRatio,
		task.TargetWidth,
		task.TargetHeight,
		task.UpstreamSize,
		task.OutputFormat,
		task.ContentType,
		task.ChannelMappedModel,
		string(channelUsageFields),
		task.RequestPayloadHash,
		task.UserAgent,
		task.IPAddress,
		task.InboundEndpoint,
		createdAt,
	)
	return err
}

func (r *openAIImageTaskRepository) GetTask(ctx context.Context, id string) (*service.OpenAIImageTask, error) {
	if r == nil || r.sql == nil {
		return nil, nil
	}
	query := `
		SELECT
			id, status, progress, request_body, api_key_id, user_id, group_id, account_id,
			model, model_alias, image_tier, aspect_ratio, target_width, target_height,
			upstream_size, output_format, content_type, channel_mapped_model,
			channel_usage_fields, request_payload_hash, user_agent, ip_address,
			inbound_endpoint, result_json, error_code, error_message,
			created_at, started_at, completed_at
		FROM image_tasks
		WHERE id = $1
	`
	task, err := r.scanTask(ctx, query, id)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return task, nil
}

func (r *openAIImageTaskRepository) ClaimTask(ctx context.Context, id string, staleRunningAfterSeconds int64) (*service.OpenAIImageTask, error) {
	if r == nil || r.sql == nil {
		return nil, nil
	}
	if staleRunningAfterSeconds <= 0 {
		staleRunningAfterSeconds = openAIImageTaskDefaultStaleSeconds()
	}
	query := `
		WITH next AS (
			SELECT id
			FROM image_tasks
			WHERE id = $1
				AND (
					status = $2
					OR (
						status = $3
						AND started_at IS NOT NULL
						AND started_at < NOW() - ($4 * interval '1 second')
					)
				)
			FOR UPDATE SKIP LOCKED
		)
		UPDATE image_tasks AS tasks
		SET status = $5,
			progress = 10,
			started_at = NOW(),
			completed_at = NULL,
			error_code = NULL,
			error_message = NULL,
			updated_at = NOW(),
			retry_count = CASE WHEN tasks.status = $3 THEN tasks.retry_count + 1 ELSE tasks.retry_count END
		FROM next
		WHERE tasks.id = next.id
		RETURNING
			tasks.id, tasks.status, tasks.progress, tasks.request_body, tasks.api_key_id, tasks.user_id,
			tasks.group_id, tasks.account_id, tasks.model, tasks.model_alias, tasks.image_tier,
			tasks.aspect_ratio, tasks.target_width, tasks.target_height, tasks.upstream_size,
			tasks.output_format, tasks.content_type, tasks.channel_mapped_model,
			tasks.channel_usage_fields, tasks.request_payload_hash, tasks.user_agent,
			tasks.ip_address, tasks.inbound_endpoint, tasks.result_json, tasks.error_code,
			tasks.error_message, tasks.created_at, tasks.started_at, tasks.completed_at
	`
	task, err := r.scanTask(
		ctx,
		query,
		id,
		service.OpenAIImageTaskStatusQueued,
		service.OpenAIImageTaskStatusInProgress,
		staleRunningAfterSeconds,
		service.OpenAIImageTaskStatusInProgress,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return task, nil
}

func (r *openAIImageTaskRepository) ClaimNextPendingTask(ctx context.Context, staleRunningAfterSeconds int64) (*service.OpenAIImageTask, error) {
	if r == nil || r.sql == nil {
		return nil, nil
	}
	if staleRunningAfterSeconds <= 0 {
		staleRunningAfterSeconds = openAIImageTaskDefaultStaleSeconds()
	}
	query := `
		WITH next AS (
			SELECT id
			FROM image_tasks
			WHERE status = $1
				OR (
					status = $2
					AND started_at IS NOT NULL
					AND started_at < NOW() - ($3 * interval '1 second')
				)
			ORDER BY created_at ASC
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		UPDATE image_tasks AS tasks
		SET status = $4,
			progress = 10,
			started_at = NOW(),
			completed_at = NULL,
			error_code = NULL,
			error_message = NULL,
			updated_at = NOW(),
			retry_count = CASE WHEN tasks.status = $2 THEN tasks.retry_count + 1 ELSE tasks.retry_count END
		FROM next
		WHERE tasks.id = next.id
		RETURNING
			tasks.id, tasks.status, tasks.progress, tasks.request_body, tasks.api_key_id, tasks.user_id,
			tasks.group_id, tasks.account_id, tasks.model, tasks.model_alias, tasks.image_tier,
			tasks.aspect_ratio, tasks.target_width, tasks.target_height, tasks.upstream_size,
			tasks.output_format, tasks.content_type, tasks.channel_mapped_model,
			tasks.channel_usage_fields, tasks.request_payload_hash, tasks.user_agent,
			tasks.ip_address, tasks.inbound_endpoint, tasks.result_json, tasks.error_code,
			tasks.error_message, tasks.created_at, tasks.started_at, tasks.completed_at
	`
	task, err := r.scanTask(
		ctx,
		query,
		service.OpenAIImageTaskStatusQueued,
		service.OpenAIImageTaskStatusInProgress,
		staleRunningAfterSeconds,
		service.OpenAIImageTaskStatusInProgress,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return task, nil
}

func (r *openAIImageTaskRepository) MarkTaskSucceeded(ctx context.Context, id string, accountID *int64, result map[string]any) error {
	if r == nil || r.sql == nil {
		return nil
	}
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("marshal image task result: %w", err)
	}
	query := `
		UPDATE image_tasks
		SET status = $2,
			progress = 100,
			account_id = $3,
			result_json = $4::jsonb,
			error_code = NULL,
			error_message = NULL,
			completed_at = NOW(),
			updated_at = NOW()
		WHERE id = $1
	`
	_, err = r.sql.ExecContext(ctx, query, id, service.OpenAIImageTaskStatusCompleted, accountID, string(resultJSON))
	return err
}

func (r *openAIImageTaskRepository) MarkTaskFailed(ctx context.Context, id, code, message string) error {
	if r == nil || r.sql == nil {
		return nil
	}
	query := `
		UPDATE image_tasks
		SET status = $2,
			progress = 100,
			error_code = $3,
			error_message = $4,
			completed_at = NOW(),
			updated_at = NOW()
		WHERE id = $1
	`
	_, err := r.sql.ExecContext(ctx, query, id, service.OpenAIImageTaskStatusFailed, code, message)
	return err
}

func (r *openAIImageTaskRepository) scanTask(ctx context.Context, query string, args ...any) (*service.OpenAIImageTask, error) {
	task := &service.OpenAIImageTask{Object: "image.task"}
	var (
		requestBody        []byte
		channelUsageFields []byte
		resultJSON         []byte
		apiKeyID           sql.NullInt64
		userID             sql.NullInt64
		groupID            sql.NullInt64
		accountID          sql.NullInt64
		errorCode          sql.NullString
		errorMessage       sql.NullString
		createdAt          time.Time
		startedAt          sql.NullTime
		completedAt        sql.NullTime
	)
	if err := scanSingleRow(
		ctx,
		r.sql,
		query,
		args,
		&task.ID,
		&task.Status,
		&task.Progress,
		&requestBody,
		&apiKeyID,
		&userID,
		&groupID,
		&accountID,
		&task.Model,
		&task.ModelAlias,
		&task.ImageTier,
		&task.AspectRatio,
		&task.TargetWidth,
		&task.TargetHeight,
		&task.UpstreamSize,
		&task.OutputFormat,
		&task.ContentType,
		&task.ChannelMappedModel,
		&channelUsageFields,
		&task.RequestPayloadHash,
		&task.UserAgent,
		&task.IPAddress,
		&task.InboundEndpoint,
		&resultJSON,
		&errorCode,
		&errorMessage,
		&createdAt,
		&startedAt,
		&completedAt,
	); err != nil {
		return nil, err
	}
	task.RequestBody = append([]byte(nil), requestBody...)
	if apiKeyID.Valid {
		task.APIKeyID = apiKeyID.Int64
	}
	if userID.Valid {
		task.UserID = userID.Int64
	}
	if groupID.Valid {
		v := groupID.Int64
		task.GroupID = &v
	}
	if accountID.Valid {
		v := accountID.Int64
		task.AccountID = &v
	}
	if len(channelUsageFields) > 0 {
		_ = json.Unmarshal(channelUsageFields, &task.ChannelUsageFields)
	}
	if len(resultJSON) > 0 {
		var result map[string]any
		if err := json.Unmarshal(resultJSON, &result); err != nil {
			return nil, fmt.Errorf("parse image task result: %w", err)
		}
		task.Result = result
	}
	if errorCode.Valid || errorMessage.Valid {
		task.Error = map[string]string{
			"code":    errorCode.String,
			"message": errorMessage.String,
		}
	}
	task.CreatedAt = createdAt.Unix()
	if startedAt.Valid {
		v := startedAt.Time.Unix()
		task.StartedAt = &v
	}
	if completedAt.Valid {
		v := completedAt.Time.Unix()
		task.CompletedAt = &v
	}
	if task.TargetWidth > 0 && task.TargetHeight > 0 {
		task.TargetSize = fmt.Sprintf("%dx%d", task.TargetWidth, task.TargetHeight)
	}
	return task, nil
}

func normalizeJSONBBytes(body []byte) []byte {
	if len(body) == 0 || !json.Valid(body) {
		return []byte("{}")
	}
	return body
}

func nullInt64Arg(value int64) any {
	if value <= 0 {
		return nil
	}
	return value
}

func openAIImageTaskDefaultStaleSeconds() int64 {
	return 30 * 60
}
