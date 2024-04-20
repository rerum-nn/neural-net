#pragma once

#include "../LossFunctions/LossFunction.h"
#include "../Network.h"
#include "../Types.h"

#include <memory>
#include <vector>

namespace neural_net {

using Optimizer = std::function<void(Network&, const Matrix&, const Matrix&, const LossFunction&, size_t)>;

class Optimizers {
public:
    static Optimizer SGD(double lr = 0.01);
    static Optimizer Momentum(double lr = 0.01, double momentum = .0);

    static Optimizer AdaGrad(double lr = 0.01);
    static Optimizer RMSProp(double lr = 0.001, double rho = 0.9);
    static Optimizer Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999);
};

// class Optimizer {
// private:
//     class OptimizerConcept;
//
// public:
//     template <typename OptimizerT>
//     Optimizer(OptimizerT&& optimizer)
//         : object_(
//               std::make_unique<OptimizerModel<OptimizerT>>(std::forward<OptimizerT>(optimizer)))
//               {
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
//         virtual void Optimize(Network& network, const Matrix& input_data, const Matrix& labels,
//                               const LossFunction& loss) = 0;
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
//         OptimizerModel(const OptimizerT& optimizer) : optimizer_(optimizer) {
//         }
//
//         OptimizerModel(OptimizerT&& optimizer) : optimizer_(std::move(optimizer)) {
//         }
//
//         void Optimize(Network& network, const Matrix& input_data, const Matrix& labels,
//                       const LossFunction& loss) override {
//             return optimizer_.Optimize(network, input_data, labels, loss);
//         }
//
//     private:
//         std::unique_ptr<OptimizerConcept> Clone() const override {
//             return std::make_unique<OptimizerModel>(optimizer_);
//         }
//
//         OptimizerT optimizer_;
//     };
//
//     std::unique_ptr<OptimizerConcept> object_;
// };

//class Optimizer {
//public:
//
//
//private:
//    Optimizer(
//        std::function<void(Network&, const Matrix&, const Matrix&, const LossFunction&, size_t)>);
//
//    std::function<void(Network&, const Matrix&, const Matrix&, const LossFunction&, size_t)>
//        optimize_;
//};

}  // namespace neural_net
