#pragma once

#include "../LinearPrimitives.h"

namespace neural_net {

class ActivationFunction {
private:
    class ActivationConcept;

public:
    template <typename ActivationT>
    ActivationFunction(ActivationT func)
        : object_(std::make_unique<ActivationModel<ActivationT>>(std::move(func))) {
    }

    ActivationFunction(const ActivationFunction& other) : object_(other.object_->Clone()) {
    }
    ActivationFunction& operator=(const ActivationFunction& other) {
        other.object_->Clone().swap(object_);
        return *this;
    }

    const ActivationConcept* operator->() const {
        return object_.get();
    }

    ActivationConcept* operator->() {
        return object_.get();
    }

private:
    class ActivationConcept {
    public:
        virtual ~ActivationConcept() = default;

        virtual void Apply(DataVector* data_vector) const = 0;
        virtual WeightMatrix Derivative(const DataVector& values) const = 0;
        virtual std::unique_ptr<ActivationConcept> Clone() const = 0;
    };

    template <typename ActivationT>
    class ActivationModel : public ActivationConcept {
    public:
        ActivationModel(ActivationT func) : func_(std::move(func)) {
        }

        void Apply(DataVector* data_vector) const override {
            func_.Apply(data_vector);
        }
        WeightMatrix Derivative(const DataVector& values) const override {
            return func_.Derivative(values);
        }
        std::unique_ptr<ActivationConcept> Clone() const override {
            return std::make_unique<ActivationModel>(*this);
        }

    private:
        ActivationT func_;
    };

    std::unique_ptr<ActivationConcept> object_;
};

}  // namespace neural_net
