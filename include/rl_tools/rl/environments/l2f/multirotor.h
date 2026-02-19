#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_MULTIROTOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_MULTIROTOR_H

#include "../../../utils/generic/typing.h"

#include "../environments.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template <typename T_T, typename T_TI, T_TI T_N, typename T_REWARD_FUNCTION>
    struct ParametersBaseSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI N = T_N;
        using REWARD_FUNCTION = T_REWARD_FUNCTION;
    };
    namespace parameters{
        template <typename T, typename TI, TI N>
        struct Dynamics{
            struct ActionLimit{
                T min;
                T max;
            };
            T rotor_positions[N][3];
            T rotor_thrust_directions[N][3];
            T rotor_torque_directions[N][3];
            T rotor_thrust_coefficients[N][3];
            T rotor_torque_constants[N];
            T rotor_time_constants_rising[N];
            T rotor_time_constants_falling[N];
            T mass;
            T gravity[3];
            T J[3][3];
            T J_inv[3][3];
            T hovering_throttle_relative; 
            ActionLimit action_limit;
        };
        template <typename T>
        struct Initialization{
            T guidance;
            T max_position;
            T max_angle;
            T max_linear_velocity;
            T max_angular_velocity;
            bool relative_rpm; 
            T min_rpm; 
            T max_rpm; 
            // fixed_mode ВИДАЛЕНО
        };
        template <typename T>
        struct Termination{
            bool enabled = false;
            T position_threshold;
            T linear_velocity_threshold;
            T angular_velocity_threshold;
            T position_integral_threshold;
            T orientation_integral_threshold;
        };
        template <typename T>
        struct ObservationNoise{
            T position;
            T orientation;
            T linear_velocity;
            T angular_velocity;
            T imu_acceleration;
        };
        template <typename T>
        struct ActionNoise{
            T normalized_rpm; 
        };
        template <typename T>
        struct Integration{
            T dt;
        };
        template <typename T_SPEC>
        struct MDP{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using Initialization = parameters::Initialization<T>;
            using ObservationNoise = parameters::ObservationNoise<T>;
            using ActionNoise = parameters::ActionNoise<T>;
            using Termination = parameters::Termination<T>;
            using REWARD_FUNCTION = typename SPEC::REWARD_FUNCTION;
            Initialization init;
            REWARD_FUNCTION reward;
            ObservationNoise observation_noise;
            ActionNoise action_noise;
            Termination termination;
        };
        template <typename T>
        struct Disturbances{
            struct UnivariateGaussian{
                T mean;
                T std;
            };
            UnivariateGaussian random_force;
            UnivariateGaussian random_torque;
        };
        template <typename T_T>
        struct DomainRandomization{ 
            using T = T_T;
            T thrust_to_weight_min; 
            T thrust_to_weight_max;
            T torque_to_inertia_min; 
            T torque_to_inertia_max;
            T mass_min; 
            T mass_max;
            T mass_size_deviation; 
            T rotor_time_constant_rising_min; 
            T rotor_time_constant_rising_max;
            T rotor_time_constant_falling_min;
            T rotor_time_constant_falling_max;
            T rotor_torque_constant_min; 
            T rotor_torque_constant_max;
            T orientation_offset_angle_max;
            T disturbance_force_max; 
        };
        template <typename T>
        static constexpr DomainRandomization<T> domain_randomization_disabled = {};
        template <typename T, typename TI>
        struct Trajectory{
            static constexpr TI MIXTURE_N = 2; 
            T mixture[MIXTURE_N];
            struct Langevin{
                T gamma;
                T omega;
                T sigma;
                T alpha;
            };
            Langevin langevin;
        };
    }
    template <typename T_SPEC>
    struct ParametersBase{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI N = SPEC::N;
        using Dynamics = parameters::Dynamics<T, TI, N>;
        using Integration = parameters::Integration<T>;
        using MDP = parameters::MDP<SPEC>;
        Dynamics dynamics;
        Integration integration;
        MDP mdp;
    };
    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct ParametersSpecification{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };

    template <typename SPEC>
    struct ParametersDisturbances: SPEC::NEXT_COMPONENT{
        static constexpr typename SPEC::TI N = SPEC::NEXT_COMPONENT::N;
        using Disturbances = parameters::Disturbances<typename SPEC::T>;
        Disturbances disturbances;
    };

    struct DefaultParametersDomainRandomizationOptions{
        static constexpr bool THRUST_TO_WEIGHT = false;
        static constexpr bool MASS = false;
        static constexpr bool TORQUE_TO_INERTIA = false;
        static constexpr bool MASS_SIZE_DEVIATION = false;
        static constexpr bool ROTOR_TORQUE_CONSTANT = false;
        static constexpr bool DISTURBANCE_FORCE = false;
        static constexpr bool ROTOR_TIME_CONSTANT = false;
    };
    template <typename T_T, typename T_TI, typename T_OPTIONS, typename T_NEXT_COMPONENT>
    struct ParametersDomainRandomizationSpecification{
        using T = T_T;
        using TI = T_TI;
        using DOMAIN_RANDOMIZATION_OPTIONS = T_OPTIONS;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename SPEC>
    struct ParametersDomainRandomization: SPEC::NEXT_COMPONENT{
        static constexpr typename SPEC::TI N = SPEC::NEXT_COMPONENT::N;
        using DomainRandomization = parameters::DomainRandomization<typename SPEC::T>;
        DomainRandomization domain_randomization;
    };
    struct DefaultParametersTrajectoryOptions{
        static constexpr bool LANGEVIN = false;
    };
    template <typename T_T, typename T_TI, typename T_OPTIONS, typename T_NEXT_COMPONENT>
    struct ParametersTrajectorySpecification{
        using T = T_T;
        using TI = T_TI;
        using TRAJECTORY_OPTIONS = T_OPTIONS;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct ParametersTrajectory: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        static constexpr typename SPEC::TI N = SPEC::NEXT_COMPONENT::N;
        using Trajectory = parameters::Trajectory<typename SPEC::T, typename SPEC::TI>;
        Trajectory trajectory;
    };

    namespace observation{
        template <typename T_TI>
        struct LastComponent{
            static constexpr T_TI CURRENT_DIM = 0;
            static constexpr T_TI DIM = 0;
        };
        template <typename T_TI, bool T_ENABLE, typename T_CURRENT_COMPONENT, typename T_NEXT_COMPONENT=LastComponent<T_TI>>
        struct MultiplexSpecification{
            using TI = T_TI;
            static constexpr bool ENABLE = T_ENABLE;
            using CURRENT_COMPONENT = T_CURRENT_COMPONENT;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct Multiplex{
            using TI = typename SPEC::TI;
            static constexpr bool ENABLE = SPEC::ENABLE;
            using CURRENT_COMPONENT = typename SPEC::CURRENT_COMPONENT;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = ENABLE ? CURRENT_COMPONENT::CURRENT_DIM : 0;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_TI>
        struct NONE{
            static constexpr T_TI DIM = 0;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PositionSpecificationPrivileged: PositionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct Position{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationQuaternionSpecificationPrivileged: OrientationQuaternionSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationQuaternion{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct OrientationRotationMatrixSpecificationPrivileged: OrientationRotationMatrixSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct OrientationRotationMatrix{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 9;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct LinearVelocitySpecificationPrivileged: LinearVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct LinearVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocitySpecificationPrivileged: AngularVelocitySpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct AngularVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct IMUAccelerometerSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct IMUAccelerometerSpecificationPrivileged: IMUAccelerometerSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct IMUAccelerometer{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct MagnetometerSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct MagnetometerSpecificationPrivileged: IMUAccelerometerSpecification<T_T, T_TI, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct Magnetometer{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 2;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_DELAY, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocityDelayedSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr TI DELAY = T_DELAY;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename T_T, typename T_TI, T_TI T_DELAY, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct AngularVelocityDelayedSpecificationPrivileged: AngularVelocityDelayedSpecification<T_T, T_TI, T_DELAY, T_NEXT_COMPONENT>{
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct AngularVelocityDelayed{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct PoseIntegralSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = true;
        };
        template <typename SPEC>
        struct PoseIntegral{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr bool PRIVILEGED = SPEC::PRIVILEGED;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 3 + 1;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };

        // FlightModeSpecification ТА FlightMode ВИДАЛЕНО

        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RotorSpeedsSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RotorSpeeds{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 4;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ActionHistorySpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        };
        template <typename SPEC>
        struct ActionHistory{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI HISTORY_LENGTH = SPEC::HISTORY_LENGTH;
            static constexpr TI ACTION_DIM = 4;
            static constexpr TI CURRENT_DIM = ACTION_DIM * HISTORY_LENGTH;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct RandomForceSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct RandomForce{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 6;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_N, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ParametersMotorPositionSpecification {
            using T = T_T;
            using TI = T_TI;
            static constexpr TI N = T_N;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct ParametersMotorPosition{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = SPEC::N * 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, T_TI T_N, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ParametersThrustCurvesSpecification {
            using T = T_T;
            using TI = T_TI;
            static constexpr TI N = T_N;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct ParametersThrustCurves{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = SPEC::N * 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ParametersMassSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct ParametersMass{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 1;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct ParametersInertiaSpecification {
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
        };
        template <typename SPEC>
        struct ParametersInertia{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 9;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct TrajectoryTrackingPositionSpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename SPEC>
        struct TrajectoryTrackingPosition{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
        template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = LastComponent<T_TI>>
        struct TrajectoryTrackingLinearVelocitySpecification{
            using T = T_T;
            using TI = T_TI;
            using NEXT_COMPONENT = T_NEXT_COMPONENT;
            static constexpr bool PRIVILEGED = false;
        };
        template <typename SPEC>
        struct TrajectoryTrackingLinearVelocity{
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
            static constexpr TI CURRENT_DIM = 3;
            static constexpr TI DIM = NEXT_COMPONENT::DIM + CURRENT_DIM;
        };
    }

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT = void>
    struct StateSpecification{
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };

    template <typename T_SPEC>
    struct StateBase{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI DIM = 13;
        T position[3];
        T orientation[4];
        T linear_velocity[3];
        T angular_velocity[3];
        // mode ВИДАЛЕНО
    };
    template <typename T_SPEC>
    struct StateLastAction: T_SPEC::NEXT_COMPONENT{ 
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI DIM = ACTION_DIM + NEXT_COMPONENT::DIM;
        T last_action[ACTION_DIM];
    };
    template <typename T_SPEC>
    struct StateLinearAcceleration: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        T linear_acceleration[3];
    };

    template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, typename T_NEXT_COMPONENT>
    struct StateAngularVelocityDelaySpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct StateAngularVelocityDelay: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 3 + NEXT_COMPONENT::DIM;
        static constexpr TI HISTORY_MEM_LENGTH = SPEC::HISTORY_LENGTH == 0 ? 1 : SPEC::HISTORY_LENGTH;
        T angular_velocity_history[HISTORY_MEM_LENGTH][3];
    };

    template <typename T_T, typename T_TI, bool T_CLOSED_FORM = false, typename T_NEXT_COMPONENT = void>
    struct StateRotorsSpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr bool CLOSED_FORM = T_CLOSED_FORM;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    template <typename T_SPEC>
    struct StateRotors: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool CLOSED_FORM = SPEC::CLOSED_FORM;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI PARENT_DIM = NEXT_COMPONENT::DIM;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI DIM = PARENT_DIM + ACTION_DIM;
        T rpm[ACTION_DIM];
    };
    template <typename T_T, typename T_TI, T_TI T_HISTORY_LENGTH, bool T_CLOSED_FORM, typename T_NEXT_COMPONENT>
    struct StateRotorsHistorySpecification{
        using T = T_T;
        using TI = T_TI;
        static constexpr bool CLOSED_FORM = T_CLOSED_FORM;
        static constexpr TI HISTORY_LENGTH = T_HISTORY_LENGTH;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };

    template <typename T_SPEC>
    struct StateRotorsHistory: StateRotors<T_SPEC>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = StateRotors<SPEC>;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI HISTORY_LENGTH = SPEC::HISTORY_LENGTH;
        static constexpr TI PARENT_DIM = StateRotors<SPEC>::DIM;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI DIM = PARENT_DIM + HISTORY_LENGTH * ACTION_DIM;
        TI current_step;
        T action_history[HISTORY_LENGTH][4];
    };

    template <typename T_SPEC>
    struct StateRandomForce: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 6 + NEXT_COMPONENT::DIM;
        T force[3];
        T torque[3];
    };
    template <typename T_SPEC>
    struct StatePoseErrorIntegral: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = true;
        static constexpr TI DIM = 2;
        T position_integral[3];
        T orientation_integral;
    };
    template <typename T_SPEC>
    struct StateRandomOrientationOffset: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 4 + NEXT_COMPONENT::DIM;
        T orientation_offset[4];
    };
    enum TrajectoryType{
        POSITION = 0,
        LANGEVIN = 1
    };
    template <typename T_SPEC>
    struct StateTrajectory: T_SPEC::NEXT_COMPONENT{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr TI DIM = 7 + NEXT_COMPONENT::DIM;
        struct Trajectory{
            TrajectoryType type;
            struct Langevin{
                T position[3];
                T velocity[3];
                T position_raw[3];
                T velocity_raw[3];
            };
            union{
                Langevin langevin;
            };
        };
        Trajectory trajectory;
    };

    template <typename T, typename TI, TI ANGULAR_VELOCITY_HISTORY = 0>
    using DefaultState = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_HISTORY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>; 
    
    template <typename T, typename TI, TI ACTION_HISTORY_LENGTH = 1, TI ANGULAR_VELOCITY_HISTORY = 0, bool CLOSED_FORM = false>
    using DefaultActionHistoryState = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, DefaultState<T, TI, ANGULAR_VELOCITY_HISTORY>>>>>;
    
    // DefaultObservation ОНОВЛЕНО: FlightMode ПРИБРАНО З ЛАНЦЮЖКА
    template <typename T, typename TI, TI ANGULAR_VELOCITY_DELAY=0, typename NEXT_OBSERVATION = observation::LastComponent<TI>>
    using DefaultObservation = observation::TrajectoryTrackingPosition<observation::TrajectoryTrackingPositionSpecification<T, TI, observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI, observation::TrajectoryTrackingLinearVelocity<observation::TrajectoryTrackingLinearVelocitySpecification<T, TI, observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY, NEXT_OBSERVATION>>>>>>>>;
    
    template <typename T, typename TI, TI ACTION_HISTORY_LENGTH, TI ANGULAR_VELOCITY_DELAY=0, typename NEXT_OBSERVATION = observation::LastComponent<TI>>
    using DefaultActionHistoryObservation = DefaultObservation<T, TI, ANGULAR_VELOCITY_DELAY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, NEXT_OBSERVATION>>>;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "./parameters/default.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    template <typename T_T, typename T_TI, typename T_PARAMETERS = typename parameters::DEFAULT_PARAMETERS_FACTORY<T_T, T_TI>::STATIC_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using STATIC_PARAMETERS = T_PARAMETERS;
        using STATE_TYPE = typename STATIC_PARAMETERS::STATE_TYPE;
        using OBSERVATION_TYPE = typename STATIC_PARAMETERS::OBSERVATION_TYPE;
        using OBSERVATION_TYPE_PRIVILEGED = typename STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
        using PARAMETERS = typename STATIC_PARAMETERS::PARAMETERS;
        static constexpr auto PARAMETER_VALUES = STATIC_PARAMETERS::PARAMETER_VALUES;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    template <typename T_SPEC>
    struct Multirotor{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = typename SPEC::PARAMETERS;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI EPISODE_STEP_LIMIT = SPEC::STATIC_PARAMETERS::EPISODE_STEP_LIMIT;

        static constexpr TI ACTION_HISTORY_LENGTH = SPEC::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH;

        using State = typename SPEC::STATE_TYPE;
        using Observation = typename SPEC::OBSERVATION_TYPE;
        using ObservationPrivileged = typename SPEC::OBSERVATION_TYPE_PRIVILEGED;

        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;
        static constexpr bool PRIVILEGED_OBSERVATION_AVAILABLE = !rl_tools::utils::typing::is_same_v<typename SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED, l2f::observation::NONE<TI>>;
        Parameters parameters = SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
    };
    namespace l2f{
        template <typename T_T, typename T_TI, typename T_PARAMETERS, bool T_SAMPLE_INITIAL_PARAMETERS = true>
        struct MultiTaskSpecification: Specification<T_T, T_TI, T_PARAMETERS>{
            static constexpr bool SAMPLE_INITIAL_PARAMETERS = T_SAMPLE_INITIAL_PARAMETERS;
        };
    }
    template <typename T_SPEC>
    struct MultirotorMultiTask: Multirotor<T_SPEC>{ /* just a tag for dispatch */ };
    template <typename SPEC>
    struct PREVENT_DEFAULT_GET_UI<MultirotorMultiTask<SPEC>> : rl_tools::utils::typing::true_type {};
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "parameters/registry.h"

#endif