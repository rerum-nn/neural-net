#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Sequential.h"
#include "../Types.h"

#include <any>
#include <vector>

namespace neural_net {

using Optimizer =
    std::function<void(Sequential&, const Matrix&, const Matrix&, const LossFunction&, size_t)>;

// class Optimizer {
// private:
//     class OptimizerConcept;
//
// public:
//     template <typename OptimizerT>
//     Optimizer(OptimizerT&& layer)
//         : object_(std::make_unique<OptimizerModel<OptimizerT>>(std::forward<OptimizerT>(layer)))
//         {
//     }
//
//     Optimizer(const Optimizer& other) : object_(other ? other.object_->Clone() : nullptr) {
//     }
//
//     Optimizer& operator=(const Optimizer& other) {
//         return *this = Optimizer(other);
//     }
//
//     Optimizer(Optimizer&&) noexcept = default;
//     Optimizer& operator=(Optimizer&&) noexcept = default;
//
//     const OptimizerConcept* operator->() const {
//         return object_.get();
//     }
//
//     OptimizerConcept* operator->() {
//         return object_.get();
//     }
//
//     operator bool() const {
//         return object_.operator bool();
//     }
//
// private:
//     class OptimizerConcept {
//     public:
//         virtual void InitParameters(const std::vector<Layer>& layers) = 0;
//         virtual void Update(const std::vector<ParametersGrad>& params, size_t layer_id) = 0;
//         virtual void BatchCallback() = 0;
//         virtual void EpochCallback(size_t epoch, size_t max_epoch) = 0;
//
//         virtual ~OptimizerConcept() = default;
//
//     private:
//         virtual std::unique_ptr<OptimizerConcept> Clone() const = 0;
//
//         friend class Optimizer;
//     };
//
//     template <typename OptimizerT>
//     class OptimizerModel : public OptimizerConcept {
//     public:
//         OptimizerModel(const OptimizerT& func) : layer_(func) {
//         }
//
//         OptimizerModel(OptimizerT&& func) : layer_(std::move(func)) {
//         }
//
//         void InitParameters(const std::vector<Layer>& layers) override {
//             layer_->InitParameters(layers);
//         }
//         void Update(const std::vector<ParametersGrad>& params, size_t layer_id) override {
//             layer_->Update(params, layer_id);
//         }
//         void BatchCallback() override {
//             layer_->BatchCallback();
//         }
//         void EpochCallback(size_t epoch, size_t max_epoch) override {
//             layer_->EpochCallback(epoch, max_epoch);
//         }
//
//     private:
//         std::unique_ptr<OptimizerConcept> Clone() const override {
//             return std::make_unique<OptimizerModel>(layer_);
//         }
//
//         OptimizerT layer_;
//     };
//
//     std::unique_ptr<OptimizerConcept> object_;
// };

class Optimizers {
public:
    static Optimizer SGD(double lr = 0.01);
    static Optimizer Momentum(double lr = 0.01, double momentum = .0);

    static Optimizer AdaGrad(double lr = 0.01);
    static Optimizer RMSProp(double lr = 0.001, double rho = 0.9);
    static Optimizer Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999);
};

}  // namespace neural_net
