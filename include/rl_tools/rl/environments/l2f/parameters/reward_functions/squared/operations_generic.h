#include "../../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_SQUARED_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_REWARD_FUNCTIONS_SQUARED_OPERATIONS_GENERIC_H

#include "../../../multirotor.h"
#include <rl_tools/utils/generic/typing.h>
#include <rl_tools/utils/generic/vector_operations.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f::parameters::reward_functions{
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward_components(DEVICE& device, const Multirotor<SPEC>& env, const typename Multirotor<SPEC>::Parameters& parameters, const Squared<T>& reward_parameters, const StateBase<STATE_SPEC>& state_dispatch, const STATE& state, const Matrix<ACTION_SPEC>& action,  const STATE& next_state, typename Squared<T>::Components& components, RNG& rng){
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        STATE desired_state;
        
        // Отримуємо цільову позицію і орієнтацію
        get_desired_state(device, env, parameters, state, desired_state, rng);

        int mode = (int)state.mode;

        // --- 1. Orientation Logic ---
        
        T qw = state.orientation[0];
        T qx = state.orientation[1];
        T qy = state.orientation[2];
        T qz = state.orientation[3];
        
        T t_qw = desired_state.orientation[0];
        T t_qx = desired_state.orientation[1];
        T t_qy = desired_state.orientation[2];
        T t_qz = desired_state.orientation[3];
        
        // A. Повна помилка (Yaw + Pitch + Roll)
        // Використовується, коли нам важливо, куди дивиться ніс дрона.
        T q_dot = t_qw*qw + t_qx*qx + t_qy*qy + t_qz*qz;
        T abs_q_dot = math::abs(device.math, q_dot);
        if (abs_q_dot > 1.0) abs_q_dot = 1.0;
        T full_rotation_error = 2 * math::acos(device.math, abs_q_dot);

        // B. Помилка стабілізації (Тільки Pitch + Roll)
        // Використовується, щоб дрон тримав горизонт і не падав.
        T tilt_penalty = 2 * (qx*qx + qy*qy); 
        
        if (mode == 1) {
            // MODE 1 (Rotate):
            // Дрон бачить ціль Yaw і активно повертається до неї.
            // + Tilt penalty для плоского спіну.
            components.orientation_cost = full_rotation_error + tilt_penalty;
        } 
        else {
            // MODE 0, 2 (Hover/Move):
            // Дрон "сліпий" до компасу (Yaw). Він ігнорує full_rotation_error.
            // Ми вимагаємо тільки горизонтальної стабілізації.
            components.orientation_cost = tilt_penalty;
        }

        // --- 2. Angular Velocity (Демпфер/Гальмо) ---
        
        T angular_vel_magnitude = math::sqrt(device.math, 
            math::max(device.math, (T)0, state.angular_velocity[0] * state.angular_velocity[0] + 
            state.angular_velocity[1] * state.angular_velocity[1] + 
            state.angular_velocity[2] * state.angular_velocity[2])
        );

        // Якщо Mode 0 або 2:
        // Ми збільшуємо штраф у 5 разів. Дрон буде намагатися зупинити будь-яке обертання.
        // Це змушує його "зафіксувати" той кут, на якому він зараз знаходиться.
        T angular_vel_weight_multiplier = (mode == 1) ? (T)1.0 : (T)5.0;

        components.angular_vel_cost = angular_vel_weight_multiplier * angular_vel_magnitude;


        // --- 3. Position Cost ---
        T x = state.position[0] - desired_state.position[0];
        T y = state.position[1] - desired_state.position[1];
        T z = state.position[2] - desired_state.position[2];
        components.position_cost = math::sqrt(device.math, math::max(device.math, (T)0, x*x + y*y + z*z));
        
        if(reward_parameters.position_clip > 0){
            components.position_cost = math::min(device.math, components.position_cost, reward_parameters.position_clip);
        }

        // --- 4. Velocity & Accel Costs ---
        T vx = state.linear_velocity[0] - desired_state.linear_velocity[0];
        T vy = state.linear_velocity[1] - desired_state.linear_velocity[1];
        T vz = state.linear_velocity[2] - desired_state.linear_velocity[2];
        components.linear_vel_cost = math::sqrt(device.math, math::max(device.math, (T)0, vx*vx + vy*vy + vz*vz));

        T linear_acc[3];
        T angular_acc[3];
        rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(next_state.linear_velocity, state.linear_velocity, linear_acc);
        components.linear_acc_cost = math::sqrt(device.math, math::max(device.math, (T)0, linear_acc[0] * linear_acc[0] + linear_acc[1] * linear_acc[1] + linear_acc[2] * linear_acc[2])) / parameters.integration.dt;
        rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(next_state.angular_velocity, state.angular_velocity, angular_acc);
        components.angular_acc_cost = math::sqrt(device.math, math::max(device.math, (T)0, angular_acc[0] * angular_acc[0] + angular_acc[1] * angular_acc[1] + angular_acc[2] * angular_acc[2])) / parameters.integration.dt;

        // --- 5. Action Cost ---
        T action_diff[ACTION_DIM];
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action_throttle_relative = (get(action, 0, action_i) + (T)1.0)/(T)2.0;
            action_diff[action_i] = action_throttle_relative - parameters.dynamics.hovering_throttle_relative;
        }
        components.action_cost = rl_tools::utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        components.action_cost *= components.action_cost;
    }

    // --- Template Instantiations for different state types ---

    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward_components(DEVICE& device, const Multirotor<SPEC>& env, const typename Multirotor<SPEC>::Parameters& parameters, const Squared<T>& reward_parameters, const StateLastAction<STATE_SPEC>& state_dispatch, const STATE& state, const Matrix<ACTION_SPEC>& action,  const STATE& next_state, typename Squared<T>::Components& components, RNG& rng) {
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        reward_components(device, env, parameters, reward_parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), state, action, next_state, components, rng);
        components.d_action_cost = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T d_action_value = get(action, 0, action_i) - state.last_action[action_i];
            components.d_action_cost += d_action_value * d_action_value;
        }
        components.d_action_cost = math::sqrt(device.math, components.d_action_cost);
    }

    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward_components(DEVICE& device, const Multirotor<SPEC>& env, const typename Multirotor<SPEC>::Parameters& parameters, const Squared<T>& reward_parameters, const StatePoseErrorIntegral<STATE_SPEC>& state_dispatch, const STATE& state, const Matrix<ACTION_SPEC>& action,  const STATE& next_state, typename Squared<T>::Components& components, RNG& rng){
        reward_components(device, env, parameters, reward_parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), state, action, next_state, components, rng);
        static constexpr T EPSILON = 1e-4;
        components.position_error_integral_cost = math::sqrt(device.math, next_state.position_integral[0]*next_state.position_integral[0] + next_state.position_integral[1]*next_state.position_integral[1] + next_state.position_integral[2]*next_state.position_integral[2] + EPSILON);
    }

    template<typename DEVICE, typename SPEC, typename STATE, typename ACTION_SPEC, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void log_reward(DEVICE& device, const Multirotor<SPEC>& env, const typename Multirotor<SPEC>::Parameters& parameters, const Squared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action,  const STATE& next_state, RNG& rng, typename DEVICE::index_t cadence = 1){
        auto components = reward_components(device, env, parameters, reward_parameters, state, action, next_state, rng);
        add_scalar(device, device.logger, "reward/orientation_cost", components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward/position_cost",    components.position_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_vel_cost",  components.linear_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_vel_cost", components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward/linear_acc_cost",  components.linear_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/angular_acc_cost", components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward/action_cost",      components.action_cost, cadence);
        add_scalar(device, device.logger, "reward/pre_exp",         -components.weighted_cost, cadence);

        add_scalar(device, device.logger, "reward_weighted/orientation_cost", reward_parameters.orientation          * components.orientation_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/position_cost",    reward_parameters.position             * components.position_cost,    cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_vel_cost",  reward_parameters.linear_velocity      * components.linear_vel_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_vel_cost", reward_parameters.angular_velocity     * components.angular_vel_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/linear_acc_cost" , reward_parameters.linear_acceleration  * components.linear_acc_cost,  cadence);
        add_scalar(device, device.logger, "reward_weighted/angular_acc_cost", reward_parameters.angular_acceleration * components.angular_acc_cost, cadence);
        add_scalar(device, device.logger, "reward_weighted/action_cost",      reward_parameters.action               * components.action_cost,      cadence);
        
        add_scalar(device, device.logger, "reward_share/orientation", reward_parameters.orientation          * components.orientation_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/position",    reward_parameters.position             * components.position_cost    / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_vel",  reward_parameters.linear_velocity      * components.linear_vel_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_vel", reward_parameters.angular_velocity     * components.angular_vel_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/linear_acc",  reward_parameters.linear_acceleration  * components.linear_acc_cost  / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/angular_acc", reward_parameters.angular_acceleration * components.angular_acc_cost / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/action",      reward_parameters.action               * components.action_cost      / components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward_share/const",       components.reward/reward_parameters.constant, cadence);

        add_scalar(device, device.logger, "reward/weighted_cost",        components.weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/scaled_weighted_cost", components.scaled_weighted_cost, cadence);
        add_scalar(device, device.logger, "reward/reward",               components.reward, cadence);
        add_scalar(device, device.logger, "reward/reward_zero",          components.reward == 0, cadence);
    }

    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename STATE, typename T, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T reward(DEVICE& device, const Multirotor<SPEC>& env, const typename Multirotor<SPEC>::Parameters& parameters, const Squared<T>& reward_parameters, const STATE& state, const Matrix<ACTION_SPEC>& action,  const STATE& next_state, RNG& rng){
        typename Squared<T>::Components components;
        
        // initializing optional components
        components.d_action_cost = 0;
        components.position_error_integral_cost = 0;
        reward_components(device, env, parameters, reward_parameters, state, state, action, next_state, components, rng);

        components.weighted_cost = 0;
        components.weighted_cost += reward_parameters.position * components.position_cost;
        components.weighted_cost += reward_parameters.orientation * components.orientation_cost;
        components.weighted_cost += reward_parameters.linear_velocity * components.linear_vel_cost;
        
        // --- SAFETY HACK: Захист від нульової ваги ---
        // Якщо в конфігурації вага стоїть 0, ми ставимо 0.05.
        // Це гарантує, що гальмо (angular_vel_cost) буде працювати,
        // і дрон зможе фіксувати кут у Mode 0 і 2.
        T ang_vel_weight = reward_parameters.angular_velocity;
        if(ang_vel_weight < 0.00001){
            ang_vel_weight = (T)0.05; 
        }
        components.weighted_cost += ang_vel_weight * components.angular_vel_cost;
        // ---------------------------------------------

        components.weighted_cost += reward_parameters.linear_acceleration * components.linear_acc_cost;
        components.weighted_cost += reward_parameters.angular_acceleration * components.angular_acc_cost;
        components.weighted_cost += reward_parameters.action * components.action_cost;
        components.weighted_cost += reward_parameters.d_action * components.d_action_cost;
        components.weighted_cost += reward_parameters.position_error_integral * components.position_error_integral_cost;

        bool terminated_flag = terminated(device, env, parameters, next_state, rng);
        components.scaled_weighted_cost = reward_parameters.scale * components.weighted_cost;

        if(terminated_flag){
            components.reward = reward_parameters.termination_penalty;
        }
        else{
            components.reward = -components.scaled_weighted_cost + reward_parameters.constant;
            components.reward = (components.reward > 0 || !reward_parameters.non_negative) ? components.reward : 0;
        }
        return components.reward;
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto name(DEVICE& device, const Squared<T>& reward_parameters){
        return "squared";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif