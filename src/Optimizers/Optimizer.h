#pragma once

#include "Layers/Linear.h"
#include "LossFunctions/LossFunction.h"
#include "Types.h"

#include <memory>
#include <vector>

namespace neural_net {

class Optimizer {
private:
    class OptimizerConcept;

public:
    static Optimizer SGD(double lr = 0.01);
    static Optimizer Momentum(double lr = 0.01, double momentum = .0);

    static Optimizer AdaGrad(double lr = 0.01);
    static Optimizer RMSprop(double lr = 0.001, double rho = 0.9);
    static Optimizer Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999);

    template <typename OptimizerT>
    Optimizer(OptimizerT&& layer)
        : object_(std::make_unique<OptimizerModel<OptimizerT>>(std::forward<OptimizerT>(layer))) {
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
        virtual void InitParameters(const std::vector<Linear>& layers) = 0;
        virtual void Update(const UpdatePack& params, size_t layer_id) = 0;
        virtual void BatchCallback() = 0;
        virtual void EpochCallback(size_t epoch, size_t max_epoch) = 0;

        virtual ~OptimizerConcept() = default;

    private:
        virtual std::unique_ptr<OptimizerConcept> Clone() const = 0;

        friend class Optimizer;
    };

    template <typename OptimizerT>
    class OptimizerModel : public OptimizerConcept {
    public:
        OptimizerModel(const OptimizerT& func) : optimizer_(func) {
        }

        OptimizerModel(OptimizerT&& func) : optimizer_(std::move(func)) {
        }

        void InitParameters(const std::vector<Linear>& layers) override {
            optimizer_.InitParameters(layers);
        }

        void Update(const UpdatePack& params, size_t layer_id) override {
            optimizer_.Update(params, layer_id);
        }

        void BatchCallback() override {
            optimizer_.BatchCallback();
        }

        void EpochCallback(size_t epoch, size_t max_epoch) override {
            optimizer_.EpochCallback(epoch, max_epoch);
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
