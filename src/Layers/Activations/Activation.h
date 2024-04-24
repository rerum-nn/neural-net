#pragma once

#include "Types.h"

#include <istream>
#include <memory>

namespace neural_net {

class Activation {
private:
    class ActivationConcept;

public:
    template <typename ActivationT>
    Activation(ActivationT&& activation)
        : object_(std::make_unique<ActivationModel<ActivationT>>(
              std::forward<ActivationT>(activation))) {
    }

    Activation(const Activation& other) : object_(other ? other.object_->Clone() : nullptr) {
    }

    Activation& operator=(const Activation& other) {
        return *this = Activation(other);
    }

    Activation(Activation&&) noexcept = default;
    Activation& operator=(Activation&&) noexcept = default;

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
        virtual Matrix Apply(const Matrix& data_vector) = 0;
        virtual Matrix BackPropagation(const Matrix& loss) const = 0;

        virtual void Serialize(std::ostream& os) const = 0;

        virtual ~ActivationConcept() = default;

    private:
        virtual std::unique_ptr<ActivationConcept> Clone() const = 0;

        friend class Activation;
    };

    template <typename ActivationT>
    class ActivationModel : public ActivationConcept {
    public:
        ActivationModel(const ActivationT& func) : activation_(func) {
        }

        ActivationModel(ActivationT&& func) : activation_(std::move(func)) {
        }

        Matrix Apply(const Matrix& data_vector) override {
            return activation_.Apply(data_vector);
        }

        Matrix BackPropagation(const Matrix& loss) const override {
            return activation_.BackPropagation(loss);
        }

        void Serialize(std::ostream& os) const override {
            activation_.Serialize(os);
        }

    private:
        std::unique_ptr<ActivationConcept> Clone() const override {
            return std::make_unique<ActivationModel>(activation_);
        }

        ActivationT activation_;
    };

    std::unique_ptr<ActivationConcept> object_;
};

}  // namespace neural_net
