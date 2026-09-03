// StreamMoE route-B extension: expose the host mapping of a vulkan buffer when
// it is host-visible (DEVICE_LOCAL|HOST_VISIBLE vidmem / rebar BAR1). The
// route-B expert pool reads/writes VRAM-resident expert weights straight from
// the host through this pointer. Returns nullptr for non-host-visible buffers
// (allocation fell back to pure device-local memory).
void * stmoe_vk_buffer_host_ptr(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || buffer->context == nullptr) {
        return nullptr;
    }
    auto * ctx = static_cast<ggml_backend_vk_buffer_context *>(buffer->context);
    return ctx->dev_buffer ? ctx->dev_buffer->ptr : nullptr;
}

// StreamMoE route-B extension: a fake-base pointer for byte offset `off`
// inside `buffer`. Vulkan kernels derive a tensor's buffer offset as
// (tensor->data - vk_ptr_base); a tensor shell whose buffer is `buffer` and
// whose data is this pointer therefore routes the kernel at `off` inside it.
// Keeps the fake-base convention intact without hardcoding 0x1000 in route B.
void * stmoe_vk_buffer_host_offset(ggml_backend_buffer_t buffer, size_t off) {
    (void) buffer;
    return (uint8_t *) vk_ptr_base + off;
}
