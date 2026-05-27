package service

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenAIImageTaskPreserveForwardErrorExtractsFailoverBody(t *testing.T) {
	err := openAIImageTaskPreserveForwardError(&UpstreamFailoverError{
		StatusCode:   http.StatusBadGateway,
		ResponseBody: []byte(`{"error":{"message":"upstream image queue timed out"}}`),
	})

	require.EqualError(t, err, "upstream error: 502 message=upstream image queue timed out")
}

func TestOpenAIImageTaskFinalForwardErrorPrefersLastForwardErrorOnAccountExhaustion(t *testing.T) {
	lastErr := errors.New("upstream request failed: http2: timeout awaiting response headers")

	require.EqualError(t,
		openAIImageTaskFinalForwardError(ErrNoAvailableAccounts, lastErr),
		lastErr.Error(),
	)
	require.EqualError(t,
		openAIImageTaskFinalForwardError(errors.New("image task failed after account failover"), lastErr),
		lastErr.Error(),
	)
}

func TestOpenAIImageTaskFinalForwardErrorKeepsNonSelectionError(t *testing.T) {
	lastErr := errors.New("upstream request failed: http2: timeout awaiting response headers")
	finalErr := errors.New("scheduler database failed")

	require.Equal(t, finalErr, openAIImageTaskFinalForwardError(finalErr, lastErr))
}
