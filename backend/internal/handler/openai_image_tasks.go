package handler

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"strings"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

func (h *OpenAIGatewayHandler) CreateImageTask(c *gin.Context) {
	if h.imageTaskService == nil {
		h.errorResponse(c, http.StatusServiceUnavailable, "api_error", "Image task service is not available")
		return
	}
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}
	if getGroupPlatformFromAPIKey(apiKey) != service.PlatformOpenAI {
		c.JSON(http.StatusNotFound, gin.H{
			"error": gin.H{
				"type":    "not_found_error",
				"message": "Image tasks API is not supported for this platform",
			},
		})
		return
	}

	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	if len(body) == 0 {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	parsed, err := h.gatewayService.ParseOpenAIImagesRequest(c, body)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if parsed.Stream {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "stream is not supported for image tasks")
		return
	}
	if parsed.Multipart {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "multipart image tasks are not supported")
		return
	}
	if h.billingCacheService != nil {
		subscription, _ := middleware2.GetSubscriptionFromContext(c)
		if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription); err != nil {
			status, code, message, retryAfter := billingErrorDetails(err)
			if retryAfter > 0 {
				c.Header("Retry-After", strconv.Itoa(retryAfter))
			}
			h.errorResponse(c, status, code, message)
			return
		}
	}

	channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, parsed.Model)
	upstreamForUsage := parsed.Model
	if mapped := strings.TrimSpace(channelMapping.MappedModel); mapped != "" {
		upstreamForUsage = mapped
	}
	subscription, _ := middleware2.GetSubscriptionFromContext(c)
	task, err := h.imageTaskService.CreateTask(
		c.Request.Context(),
		parsed,
		body,
		apiKey,
		subscription,
		channelMapping.MappedModel,
		channelMapping.ToUsageFields(parsed.Model, upstreamForUsage),
		service.HashUsageRequestPayload(body),
		c.GetHeader("User-Agent"),
		ip.GetClientIP(c),
		GetInboundEndpoint(c),
	)
	if err != nil {
		h.errorResponse(c, http.StatusServiceUnavailable, "api_error", err.Error())
		return
	}
	c.JSON(http.StatusAccepted, task)
}

func (h *OpenAIGatewayHandler) GetImageTask(c *gin.Context) {
	if h.imageTaskService == nil {
		h.errorResponse(c, http.StatusServiceUnavailable, "api_error", "Image task service is not available")
		return
	}
	id := strings.TrimSpace(c.Param("id"))
	if id == "" {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "task id is required")
		return
	}
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}
	task, ok, err := h.imageTaskService.GetTask(c.Request.Context(), id)
	if err != nil {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "Failed to load image task")
		return
	}
	if !ok {
		h.errorResponse(c, http.StatusNotFound, "not_found_error", "Image task not found")
		return
	}
	if task.APIKeyID != apiKey.ID {
		h.errorResponse(c, http.StatusNotFound, "not_found_error", "Image task not found")
		return
	}
	c.JSON(http.StatusOK, task)
}

func getGroupPlatformFromAPIKey(apiKey *service.APIKey) string {
	if apiKey == nil || apiKey.Group == nil {
		return ""
	}
	return apiKey.Group.Platform
}
