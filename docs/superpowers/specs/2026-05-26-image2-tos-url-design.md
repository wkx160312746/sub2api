# image2 TOS URL 输出设计

## 背景

image2 的同步接口和异步图片任务最终都会走 `OpenAIGatewayService.ForwardImages`。异步 worker 会记录这条转发链路写出的同一份 HTTP 响应体，所以只要在图片转发层做响应改写，就能同时覆盖同步返回和异步任务结果，不需要改任务存储模型。

最终接入使用火山引擎官方 TOS Go SDK，而不是 `byts-tos-main.zip` 中的 BYTS 业务签名 API。原因是你后续明确给出了 Volcengine Go SDK 方向；TOS 独立 SDK 包为 `github.com/volcengine/ve-tos-golang-sdk/v2/tos`。

- 使用 SDK `PutObjectV2` 直接把解码后的图片 bytes 上传到 TOS。
- 如果配置了 `public_base_url`，返回 `<public_base_url>/<object key>`。
- 如果没有配置公共域名且配置了读取链接过期时间，使用 SDK `PreSignedURL` 生成 GET 预签名读取 URL。

目标 bucket 固定为 `open-api`。

## 决策

使用 Volcengine TOS SDK 直连 TOS。网关会在 image2 结果写回客户端之前，把图片输出上传到 TOS。启用并配置 TOS 后，原本包含 `b64_json` 或 `data:image/...;base64,...` 的图片响应会被改写成 TOS URL 响应；未配置 TOS 时保持现有行为不变。

## 配置

在 `gateway` 配置下新增一组图片 TOS 存储配置，并按项目现有配置加载方式支持 env 覆盖：

- `gateway.image_tos.enabled`
- `gateway.image_tos.endpoint`
- `gateway.image_tos.region`
- `gateway.image_tos.access_key_id`
- `gateway.image_tos.secret_access_key`
- `gateway.image_tos.bucket`，默认 `open-api`
- `gateway.image_tos.public_base_url`，可选公共访问域名或 CDN 域名
- `gateway.image_tos.prefix`，可选对象前缀
- `gateway.image_tos.upload_url_expires_seconds`，保留兼容配置，默认 `900`
- `gateway.image_tos.read_link_expires_seconds`，可选；未配置 `public_base_url` 时用于生成 GET 预签名 URL

只有在 `enabled=true` 且必需连接参数齐全时，这个功能才生效。

## 架构

新增一个边界清楚的服务辅助模块 `TOSImageStorage`，由 OpenAI 图片转发层持有。它只负责 TOS 存储相关工作：

- 使用 endpoint、region、AK/SK 初始化 Volcengine TOS SDK client。
- 调用 `PutObjectV2` 上传解码后的图片 bytes。
- 按配置返回公共 URL 或 SDK GET 预签名 URL。
- 返回最终 URL，以及 bucket/key 等可用于日志和测试的元数据。

这个辅助模块不理解 OpenAI 响应格式。OpenAI 响应改写逻辑留在现有图片响应处理附近，因为那里已经处理 `b64_json`、data URL、流式完成事件，以及 OAuth Responses 转换后的图片载荷。

## 数据流

非流式图片响应：

1. 像现在一样读取上游响应体。
2. 检测 `data[].b64_json`，以及 `data[].url` 中的 data URL。
3. 逐张图片上传到 TOS。
4. 把每个图片 item 改成 `url: <tosReadUrl>`，并移除 `b64_json`。
5. 写出改写后的 JSON 响应。

OAuth Responses 图片转换：

1. 像现在一样把上游 Responses 输出转换成 OpenAI Images API 响应。
2. 写出响应前，复用同一个 JSON 改写 helper。
3. 异步图片任务会自动持久化改写后的响应，因为它记录的是同一条转发链路的 recorder 输出。

流式图片响应：

1. partial image 事件保持不变，客户端仍然可以展示生成进度。
2. final completed 图片事件在发送前上传最终图片。
3. completed payload 使用 `url` 返回，不再返回 base64 图片数据。

## 错误处理

如果 TOS 未启用或配置不完整，图片输出保持现状。

如果 TOS 已启用，但上传或读取链接生成失败，本次 image 请求失败，不回退返回 base64。这样启用后的契约更清楚：成功的 image2 响应一定包含可用的 TOS URL。如果既没有配置公共访问域名，也没有配置 `read_link_expires_seconds`，则视为 TOS 读取 URL 配置不完整。

返回给客户端的错误信息需要做脱敏；服务端日志保留足够上下文，用于定位失败发生在 TOS 的哪个步骤、哪个 bucket，以及已生成的对象 key。

## 测试

围绕响应改写 helper 和 TOS client 增加单元测试：

- SDK client 初始化使用 bucket `open-api`、endpoint、region、AK/SK。
- `b64_json` 图片会上传并改写为 `url`。
- `url` 字段里的 data URL 会上传并改写为普通 URL。
- 已经是普通 URL 的响应保持不变。
- TOS 未启用时响应保持不变。
- TOS 启用后上传失败会返回错误。
- 异步任务结果通过共享转发响应 helper 覆盖，或补一个聚焦测试确认 recorder 捕获前已经完成改写。

验证时先跑 OpenAI 图片 service 包的定向 Go 测试；如果改动触及共享配置或转发行为，再跑更广的 handler/service 测试。
