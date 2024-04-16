#pragma once

#include "../Types.h"

#include <memory>
#include <vector>

namespace neural_net {

class Optimizer {
private:
    class OptimizerConcept;

public:
    template <typename OptimizerT>
    Optimizer(OptimizerT&& optimizer)
        : object_(
              std::make_unique<OptimizerModel<OptimizerT>>(std::forward<OptimizerT>(optimizer))) {
    }

    Optimizer(const Optimizer& other) : object_(other ? other.object_->Clone() : nullptr) {

    }

    Optimizer& operator=(const Optimizer& other) {
        return *this = Optimizer(other);
    }

    Optimizer(Optimizer&&) noexcept = default;
    Optimizer& operator=(Optimizer&&) noexcept = default;

    const OptimizerConcept* operator->() const {
        return object_.get();
    }

    OptimizerConcept* operator->() {
        return object_.get();
    }

    operator bool() const {
        return object_.operator bool();
    }

private:
    class OptimizerConcept {
    public:
        virtual void Optimize(const std::vector<ParametersGrad>& params) = 0;

        virtual ~OptimizerConcept() = default;

    private:
        virtual std::unique_ptr<OptimizerConcept> Clone() const = 0;

        friend class Optimizer;
    };

    template <typename OptimizerT>
    class OptimizerModel : public OptimizerConcept {
    public:
        OptimizerModel(const OptimizerT& optimizer) : optimizer_(optimizer) {
        }

        OptimizerModel(OptimizerT&& optimizer) : optimizer_(std::move(optimizer)) {
        }

        void Optimize(const std::vector<ParametersGrad>& params) override {
            return optimizer_.Optimize(params);
        }

    private:
        std::unique_ptr<OptimizerConcept> Clone() const override {
            return std::make_unique<OptimizerModel>(optimizer_);
        }

        OptimizerT optimizer_;
    };

    std::unique_ptr<OptimizerConcept> object_;
};

}  // namespace neural_net
