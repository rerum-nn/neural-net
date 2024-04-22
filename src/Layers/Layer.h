#pragma once

#include "Types.h"

#include <memory>

namespace neural_net {

class Layer {
private:
    class LayerConcept;

public:
    template <typename LayerT>
    Layer(LayerT&& layer)
        : object_(std::make_unique<LayerModel<LayerT>>(std::forward<LayerT>(layer))) {
    }

    Layer(const Layer& other) : object_(other ? other.object_->Clone() : nullptr) {
    }

    Layer& operator=(const Layer& other) {
        return *this = Layer(other);
    }

    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    const LayerConcept* operator->() const {
        return object_.get();
    }

    LayerConcept* operator->() {
        return object_.get();
    }

    operator bool() const {
        return object_.operator bool();
    }

private:
    class LayerConcept {
    public:
        virtual Matrix Apply(const Matrix& data_vector) = 0;
        virtual std::vector<ParametersGrad> GetGradients(const Matrix& loss) = 0;
        virtual Matrix BackPropagation(const Matrix& loss) const = 0;

        virtual ~LayerConcept() = default;

    private:
        virtual std::unique_ptr<LayerConcept> Clone() const = 0;

        friend class Layer;
    };

    template <typename LayerT>
    class LayerModel : public LayerConcept {
    public:
        LayerModel(const LayerT& func) : layer_(func) {
        }

        LayerModel(LayerT&& func) : layer_(std::move(func)) {
        }

        Matrix Apply(const Matrix& data_vector) override {
            return layer_.Apply(data_vector);
        }

        std::vector<ParametersGrad> GetGradients(const Matrix& loss) override {
            return layer_.GetGradients(loss);
        }

        Matrix BackPropagation(const Matrix& loss) const override {
            return layer_.BackPropagation(loss);
        }

    private:
        std::unique_ptr<LayerConcept> Clone() const override {
            return std::make_unique<LayerModel>(layer_);
        }

        LayerT layer_;
    };

    std::unique_ptr<LayerConcept> object_;
};

}  // namespace neural_net
