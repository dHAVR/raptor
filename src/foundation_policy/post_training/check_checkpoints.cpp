#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>


namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#include "environment.h"
#include "../pre_training/config.h"
#include "../pre_training/options.h"
#include "helper.h"

// --- Helper Function: Find checkpoint recursively ignoring parent date folders ---
std::filesystem::path find_checkpoint_recursive(const std::filesystem::path& base_dir, const std::string& dynamics_id, const std::string& step) {
    if (!std::filesystem::exists(base_dir)) {
        return "";
    }

    // Format step to 15 digits padding with zeros (e.g., 250000 -> 000000000250000)
    std::ostringstream ss;
    ss << std::setw(15) << std::setfill('0') << step;
    std::string padded_step = ss.str();

    // Iterate recursively through experiments/
    for (const auto& entry : std::filesystem::recursive_directory_iterator(base_dir)) {
        if (entry.is_directory()) {
            // Check if this directory matches the dynamics_id (e.g., "115")
            if (entry.path().filename() == dynamics_id) {
                // Construct the expected full path
                // Structure: .../experiments/DATE/HASH_NAME/115/0000/steps/000000000250000/checkpoint.h5
                auto potential_path = entry.path() / "0000" / "steps" / padded_step / "checkpoint.h5";
                
                if (std::filesystem::exists(potential_path)) {
                    return potential_path;
                }
            }
        }
    }
    return "";
}
// ------------------------------------------------------------------------------

static constexpr bool DYNAMIC_ALLOCATION = true;
static constexpr TI NUM_EPISODES_EVAL = 100;
using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
using ACTOR_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
using ACTOR = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
using ENVIRONMENT = LOOP_CONFIG::ENVIRONMENT;


int main(){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 6;
    rlt::init(device, rng, seed);
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
    rlt::rl::environments::DummyUI ui;
    ENVIRONMENT dispatch_env;

    rlt::utils::extrack::Path checkpoint_path;
    // checkpoint_path.experiment = "2025-03-31_21-06-47";
    // checkpoint_path.experiment = "2025-04-01_13-43-13";
    // checkpoint_path.experiment = "2025-04-03_21-30-10";
    // checkpoint_path.experiment = "2025-04-04_17-00-11";
    // checkpoint_path.experiment = "2025-04-07_23-12-07";
    // checkpoint_path.experiment = "2025-04-08_23-23-52";
    checkpoint_path.experiment = "2025-04-16_20-10-58";
    checkpoint_path.name = "foundation-policy-pre-training";


    // cf like: 203; 139; 334; 31;


    std::filesystem::path dynamics_parameters_path = "src/foundation_policy/dynamics_parameters/";
    std::filesystem::path dynamics_parameter_index = "checkpoints.txt";

    // Define the base directory for experiments
    std::filesystem::path experiments_base_dir = "experiments";




    std::vector<std::tuple<std::string, ENVIRONMENT::Parameters::Dynamics>> query_dynamics;

    auto permute_rotors_px4_to_cf = [&device, &dispatch_env](const auto& dynamics){
        auto copy = dynamics;
        rlt::permute_rotors(device, dispatch_env, copy, 0, 3, 1, 2);
        return copy;
    };

    query_dynamics.emplace_back("x500", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::x500::real<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));

    std::ifstream dynamics_parameter_index_file(dynamics_parameter_index);
    if (!dynamics_parameter_index_file){
        std::cerr << "Failed to open dynamics parameter index file: " << dynamics_parameter_index << std::endl;
        return 1;
    }
    std::string dynamics_parameter_index_line;
    std::vector<std::string> dynamics_parameter_index_lines;
    while (std::getline(dynamics_parameter_index_file, dynamics_parameter_index_line)){
        if (dynamics_parameter_index_line.empty()){
            continue;
        }
        dynamics_parameter_index_lines.push_back(dynamics_parameter_index_line);
    }
    dynamics_parameter_index_file.close();
    const TI num_teachers = dynamics_parameter_index_lines.size();

    std::map<std::string, std::string> best_teacher;
    std::map<std::string, T> best_return;
#pragma omp parallel for
    for (TI teacher_i=0; teacher_i < num_teachers; ++teacher_i){
        RNG rank_rng;
        ENVIRONMENT env;
        EVALUATION_ACTOR_TYPE evaluation_actor;
        EVALUATION_ACTOR_TYPE::Buffer<DYNAMIC_ALLOCATION> eval_buffer;
        rlt::malloc(device, rank_rng);
        rlt::malloc(device, evaluation_actor);
        rlt::malloc(device, eval_buffer);

        rlt::init(device, rank_rng, seed + teacher_i);
        // load actor & critic
        auto checkpoint_info = dynamics_parameter_index_lines[dynamics_parameter_index_lines.size() - 1 - teacher_i];
        std::string checkpoint_info_str = checkpoint_info;
        auto cpp_copy = checkpoint_path;
        std::string target_dynamics_id;
        std::string target_step;

        if (checkpoint_info_str.find(',') != std::string::npos) {
            auto checkpoint_info_split = split_by_comma(checkpoint_info_str);
            target_dynamics_id = checkpoint_info_split[0]; 
            target_step = checkpoint_info_split[1];
            auto found_path = find_checkpoint_recursive(experiments_base_dir, target_dynamics_id, target_step);
            if (found_path.empty()) {
                std::cerr << "Error: Could not find checkpoint for dynamics-id: " << target_dynamics_id 
                          << " step: " << target_step << " in " << experiments_base_dir << std::endl;
                rlt::free(device, rank_rng);
                rlt::free(device, evaluation_actor);
                rlt::free(device, eval_buffer);
                continue; 
            }
            cpp_copy.checkpoint_path = found_path;
        } else {
             // Assume full path provided in checkpoints.txt
             std::filesystem::path p(checkpoint_info_str);
             if (!std::filesystem::exists(p)) {
                 if (std::filesystem::exists(std::filesystem::current_path() / p)) {
                     p = std::filesystem::current_path() / p;
                 } else {
                     std::cerr << "Error: Checkpoint file not found: " << p << std::endl;
                     rlt::free(device, rank_rng);
                     rlt::free(device, evaluation_actor);
                     rlt::free(device, eval_buffer);
                     continue;
                 }
             }
             cpp_copy.checkpoint_path = p;
             
             if (p.has_parent_path()) {
             if (p.has_parent_path()) {
                 auto id_dir_file = p.parent_path().parent_path().parent_path().parent_path();
                 target_dynamics_id = id_dir_file.filename().string();
                 /*
                 std::cout << "Original path: " << p << std::endl;
                 std::cout << "Parent: " << p.parent_path() << std::endl;
                 std::cout << "Parent 2: " << p.parent_path().parent_path() << std::endl;
                 std::cout << "Parent 3: " << p.parent_path().parent_path().parent_path() << std::endl;
                 std::cout << "Parent 4: " << p.parent_path().parent_path().parent_path().parent_path() << std::endl;
                 */
                 auto step_dir_file = p.parent_path();
                 target_step = step_dir_file.filename().string();
             }
             }
        }
        
        cpp_copy.attributes["dynamics-id"] = target_dynamics_id;
        cpp_copy.step = target_step;
        
        std::cout << "Loading teacher " << teacher_i << " (DynID: " << target_dynamics_id << ") from: " << cpp_copy.checkpoint_path << std::endl;

        try {
            if (std::filesystem::file_size(cpp_copy.checkpoint_path) == 0) {
                 throw std::runtime_error("File is empty");
            }
            auto actor_file = HighFive::File(cpp_copy.checkpoint_path.string(), HighFive::File::ReadOnly);
            rlt::load(device, evaluation_actor, actor_file.getGroup("actor"));
        } catch (const std::exception& e) {
             std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
             // For check_checkpoints, we can just skip this iteration entirely
             rlt::free(device, rank_rng);
             rlt::free(device, evaluation_actor);
             rlt::free(device, eval_buffer);
             continue;
        }

        std::ifstream dynamics_parameter_file = std::ifstream(dynamics_parameters_path / (cpp_copy.attributes["dynamics-id"] + ".json"));
        std::string dynamics_parameter_json((std::istreambuf_iterator<char>(dynamics_parameter_file)), std::istreambuf_iterator<char>());
        dynamics_parameter_file.close();
        rlt::from_json(device, env, dynamics_parameter_json, env.parameters);

        using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;
        using RESULT = rlt::rl::utils::evaluation::Result<RESULT_SPEC>;
        RESULT base_result, query_result;
        rlt::rl::utils::evaluation::NoData<rlt::rl::utils::evaluation::DataSpecification<RESULT_SPEC>> no_data;
        rlt::Mode<rlt::mode::Evaluation<>> mode;
        rlt::evaluate(device, env, ui, evaluation_actor, base_result, no_data, rank_rng, mode);
        std::cout << "Teacher env " << cpp_copy.checkpoint_path.string() << " mean return: " << base_result.returns_mean << " episode length: " << base_result.episode_length_mean << " share terminated: " << base_result.share_terminated << std::endl;

        for (const auto& [name, dynamics] : query_dynamics){
            env.parameters.dynamics = dynamics;
            rlt::evaluate(device, env, ui, evaluation_actor, query_result, no_data, rank_rng, mode);
            std::cout << "    " << name << " mean return: " << query_result.returns_mean << " episode length: " << query_result.episode_length_mean << " share terminated: " << query_result.share_terminated << std::endl;
#pragma omp critical
            {
                if (best_teacher.find(name) == best_teacher.end()){
                    best_teacher[name] = cpp_copy.checkpoint_path.string();
                    best_return[name] = query_result.returns_mean;
                }
                if (query_result.returns_mean > best_return[name]){
                    best_teacher[name] = cpp_copy.checkpoint_path.string();
                    best_return[name] = query_result.returns_mean;
                }
            }
        }
        rlt::free(device, rank_rng);
        rlt::free(device, evaluation_actor);
        rlt::free(device, eval_buffer);
    }
    for (const auto& [name, dynamics] : query_dynamics){
        std::cout << "Best teacher for " << name << ": " << best_teacher[name] << " with return: " << best_return[name] << std::endl;
    }
}
