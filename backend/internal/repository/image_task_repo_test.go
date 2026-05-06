package repository

import (
	"context"
	"database/sql"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/stretchr/testify/require"
)

func TestOpenAIImageTaskRepositoryCreateTask(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	repo := NewOpenAIImageTaskRepository(db)
	groupID := int64(2)
	task := &service.OpenAIImageTask{
		ID:                 "imgtask_test",
		Status:             service.OpenAIImageTaskStatusQueued,
		Progress:           0,
		CreatedAt:          1700000000,
		Model:              "gpt-image-2",
		ModelAlias:         "gpt-image-2-2K",
		ImageTier:          "2K",
		AspectRatio:        "16:9",
		TargetWidth:        2560,
		TargetHeight:       1440,
		UpstreamSize:       "1792x1024",
		OutputFormat:       "png",
		ContentType:        "application/json",
		RequestBody:        []byte(`{"model":"gpt-image-2-2K","prompt":"hello"}`),
		APIKeyID:           10,
		UserID:             20,
		GroupID:            &groupID,
		ChannelMappedModel: "gpt-image-2",
		RequestPayloadHash: "hash",
		UserAgent:          "ua",
		IPAddress:          "127.0.0.1",
		InboundEndpoint:    "/v1/image-tasks",
	}

	mock.ExpectExec("INSERT INTO image_tasks").
		WithArgs(
			task.ID,
			task.Status,
			task.Progress,
			string(task.RequestBody),
			task.APIKeyID,
			task.UserID,
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
			sqlmock.AnyArg(),
			task.RequestPayloadHash,
			task.UserAgent,
			task.IPAddress,
			task.InboundEndpoint,
			sqlmock.AnyArg(),
		).
		WillReturnResult(sqlmock.NewResult(1, 1))

	require.NoError(t, repo.CreateTask(context.Background(), task))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestOpenAIImageTaskRepositoryGetTaskNotFound(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	repo := NewOpenAIImageTaskRepository(db)
	mock.ExpectQuery("SELECT").WithArgs("missing").WillReturnError(sql.ErrNoRows)

	task, err := repo.GetTask(context.Background(), "missing")
	require.NoError(t, err)
	require.Nil(t, task)
	require.NoError(t, mock.ExpectationsWereMet())
}
