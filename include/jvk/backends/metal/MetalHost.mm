/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MetalHost.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::metal
{

std::unique_ptr<jvk::WindowHost> createWindowHost()
{
    return std::make_unique<WindowHost>();
}

std::shared_ptr<gpu::Backend> defaultBackend(juce::String* whyNot)
{
    auto b = Backend::acquire();
    const juce::String why = !b->usable() ? b->unavailableReason()
                           : b->isLost()  ? juce::String("the graphics device was lost") : juce::String();
    if (why.isEmpty()) return b;
    if (whyNot != nullptr) *whyNot = why;
    return nullptr;
}

// =============================================================================
// WindowHost — a CAMetalLayer in a native child view of a JUCE editor: the
// same layer-backed NSView the Vulkan host surfaces through MoltenVK
// (NSViewGenerator: it takes no mouse input; the editor's components sit on
// top of the pixels and get every event), here drawn by Metal itself.
// =============================================================================

std::shared_ptr<gpu::Backend> WindowHost::backend()
{
    if (backend_ == nullptr || backend_->isLost()) backend_ = Backend::acquire();
    if (!backend_->usable())
    {
        reason_ = backend_->unavailableReason();
        return nullptr;
    }
    if (backend_->isLost())
    {
        reason_ = "the graphics device was lost";
        return nullptr;
    }
    return backend_;
}

bool WindowHost::attach(juce::Component& editor, gpu::Extent sizeHint, juce::String& failure)
{
    failure = {};
    if (surface_ != nullptr) return true;
    if (backend() == nullptr) { failure = reason_; return false; }

    editor.addAndMakeVisible(view_);
    view_.setInterceptsMouseClicks(false, false);
    view_.setBounds(editor.getLocalBounds());
    nsView_ = std::make_unique<jvk::core::macos::NSViewGenerator>();
    void* nsView = nsView_->create(editor.getWidth(), editor.getHeight());
    if (nsView == nullptr) { detach(); failure = "the Metal view could not be created"; return false; }
    view_.setView(nsView);

    CALayer* layer = [(NSView*) nsView layer];
    if (layer == nil || ![layer isKindOfClass:[CAMetalLayer class]])
    {
        detach();
        failure = "the Metal layer could not be created";
        return false;
    }
    surface_ = std::make_unique<LayerSurface>(backend_, (CAMetalLayer*) layer, sizeHint);
    view_.setVisible(true);
    diag::log("metal: layer " + juce::String((int) surface_->extent().width) + "x"
              + juce::String((int) surface_->extent().height));
    return true;
}

void WindowHost::detach()
{
    surface_.reset();   // waits for its frames on the GPU
    view_.setView(nullptr);
    nsView_.reset();
    view_.setVisible(false);
    if (auto* parent = view_.getParentComponent()) parent->removeChildComponent(&view_);
}

} // namespace jvk::metal
