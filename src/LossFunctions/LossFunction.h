#pragma once

#include "../Types.h"

#include <memory>

namespace neural_net {

class LossFunction {
private:
    class LossConcept;

public:
    template <typename LossT>
    LossFunction(LossT&& loss)
        : object_(std::make_unique<LossModel<LossT>>(std::forward<LossT>(loss))) {
    }

    LossFunction(LossFunction&&) noexcept = default;
    LossFunction& operator=(LossFunction&&) noexcept = default;

    const LossConcept* operator->() const {
        return object_.get();
    }

    LossConcept* operator->() {
        return object_.get();
    }

private:
    class LossConcept {
    public:
        virtual double Loss(const Vector& present, const Vector& expected) const = 0;
        virtual Vector LossGradient(const Vector& present, const Vector& expected) const = 0;

        virtual ~LossConcept() = default;
    };

    template <typename LossT>
    class LossModel : public LossConcept {
    public:
        LossModel(const LossT& loss) : loss_(loss) {
        }

        LossModel(LossT&& loss) : loss_(std::move(loss)) {
        }

        double Loss(const Vector& present, const Vector& expected) const override {
            return loss_.Loss(present, expected);
        }

        Vector LossGradient(const Vector& present, const Vector& expected) const override {
            return loss_.LossGradient(present, expected);
        }

    private:
        LossT loss_;
    };

    std::unique_ptr<LossConcept> object_;
};

}  // namespace neural_net
