#pragma once

#include <memory>
#include <random>

// TODO: fully shit, will be replaced to EigenRand
namespace neural_net {
class RandomGenerator {
private:
    class RandomConcept;
public:
    template <typename RandomT>
    RandomGenerator(RandomT rnd)
        : object_(std::make_unique<RandomModel<RandomT>>(std::move(rnd))) {
    }

    const RandomConcept* operator->() const {
        return object_.get();
    }

    RandomConcept* operator->() {
        return object_.get();
    }

private:
    class RandomConcept {
    public:
        virtual ~RandomConcept() = default;
        virtual double Next() = 0;
    };

    template <typename RandomT>
    class RandomModel : public RandomConcept {
    public:
        RandomModel(RandomT rnd) : rnd_(std::move(rnd)) {
        }

        double Next() override {
            return rnd_.Next();
        }

    private:
        RandomT rnd_;
    };

    std::unique_ptr<RandomConcept> object_;
};

}  // namespace neural_net
