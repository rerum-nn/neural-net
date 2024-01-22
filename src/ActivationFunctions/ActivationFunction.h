#pragma once

#include "../Types.h"

#include <memory>

namespace neural_net {

class ActivationFunction {
private:
    class ActivationConcept;

public:
    template <typename ActivationT>
    ActivationFunction(ActivationT&& func)
        : object_(std::make_unique<ActivationModel<ActivationT>>(std::forward<ActivationT>(func))) {
    }

    ActivationFunction(const ActivationFunction& other)
        : object_(other ? other.object_->Clone() : nullptr) {
    }
    ActivationFunction& operator=(const ActivationFunction& other) {
        return *this = ActivationFunction(other);
    }

    ActivationFunction(ActivationFunction&&) noexcept = default;
    ActivationFunction& operator=(ActivationFunction&&) noexcept = default;

    const ActivationConcept* operator->() const {
        return object_.get();
    }

    ActivationConcept* operator->() {
        return object_.get();
    }

    operator bool() const {
        return object_.operator bool();
    }

private:
    class ActivationConcept {
    public:
        virtual Vector Apply(const Vector& data_vector) const = 0;
        virtual Matrix Derivative(const Vector& values) const = 0;

        virtual ~ActivationConcept() = default;

    private:
        virtual std::unique_ptr<ActivationConcept> Clone() const = 0;

        friend class ActivationFunction;
    };

    template <typename ActivationT>
    class ActivationModel : public ActivationConcept {
    public:
        ActivationModel(const ActivationT& func) : func_(func) {
        }
        ActivationModel(ActivationT&& func) : func_(std::move(func)) {
        }

        Vector Apply(const Vector& data_vector) const override {
            return func_.Apply(data_vector);
        }
        Matrix Derivative(const Vector& values) const override {
            return func_.Derivative(values);
        }
        std::unique_ptr<ActivationConcept> Clone() const override {
            return std::make_unique<ActivationModel>(func_);
        }

    private:
        ActivationT func_;
    };

    std::unique_ptr<ActivationConcept> object_;
};

}  // namespace neural_net
