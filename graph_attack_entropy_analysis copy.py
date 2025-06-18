import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import logging
from tqdm import tqdm
from collections import defaultdict
import traceback
# Import from your existing code
from connectivity_disruption_proposed_attack_model import (
    start_attack_proposed_model, 
    get_dataset_from_deeprobust,
    get_target_node_list,
    ProposedAttack,
    get_device
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConnectivityDisruptionExperiment:
    def __init__(self, surrogate_model='gcn', dataset='cora', defense_model='gcn'):
        self.surrogate_model = surrogate_model
        self.dataset = dataset
        self.defense_model = defense_model
        self.device = get_device()
        
        # Load dataset and get target nodes
        self.data = get_dataset_from_deeprobust(dataset=dataset)
        self.target_nodes = get_target_node_list(self.data)
        
        # Load important edge list
        with open('./important_edge_list.json', 'r') as json_file:
            important_edge_list_dict = json.load(json_file)
        self.important_edge_list = [tuple(item) for item in important_edge_list_dict[dataset]]
        
        # Initialize attack model
        self.attack_model = ProposedAttack(
            surrogate_model, dataset, defense_model, 
            self.important_edge_list
        )
        
        # Results storage
        self.experiment_results = []
        
    def compute_entropy(self, probabilities):
        """
        Compute classification entropy H(w) = -sum(p_i * log(p_i))
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1.0)
        return -np.sum(probabilities * np.log(probabilities))
    
    def compute_kl_divergence(self, p_original, p_perturbed):
        """
        Compute KL divergence KL(p||q) = sum(p_i * log(p_i/q_i))
        """
        epsilon = 1e-10
        p_original = np.clip(p_original, epsilon, 1.0)
        p_perturbed = np.clip(p_perturbed, epsilon, 1.0)
        
        return np.sum(p_original * np.log(p_original / p_perturbed))
    
    def get_node_probabilities_before_attack(self, target_node):
        """
        Get classification probabilities for target node before attack
        """
        # Get original model output
        surrogate_output = self.attack_model.surrogate_model_output
        probs_before = F.softmax(surrogate_output[target_node], dim=0)
        return probs_before.cpu().detach().numpy()
    
    def get_node_probabilities_after_attack(self, target_node, modified_adj):
        """
        Get classification probabilities for target node after attack
        """
        accuracy, pred_class, actual_class, probs_after = self.attack_model.predict(
            modified_adj, target_node
        )
        return probs_after
    
    def run_single_node_experiment(self, target_node, budget=5):
        """
        Run experiment on a single node
        """
        try:
            logger.info(f"Running experiment on node {target_node}")
            
            # Get probabilities before attack
            probs_before = self.get_node_probabilities_before_attack(target_node)
            entropy_before = self.compute_entropy(probs_before)
            
            # Get actual and predicted class before attack
            actual_class_before = int(self.data.labels[target_node])
            predicted_class_before = int(np.argmax(probs_before))
            
            # Perform attack
            modified_adj, attack_history = self.attack_model.attack(
                target_node=target_node, 
                n_perturbations=budget
            )
            
            # Get probabilities after attack
            accuracy_after, predicted_class_after, actual_class_after, probs_after = self.attack_model.predict(
                modified_adj, target_node
            )
            
            entropy_after = self.compute_entropy(probs_after)
            
            # Compute changes
            delta_entropy = entropy_after - entropy_before
            kl_divergence = self.compute_kl_divergence(probs_before, probs_after)
            
            # Check if misclassification occurred
            misclassified = (predicted_class_before != predicted_class_after) or \
                          (actual_class_before != predicted_class_after)
            
            # Determine entropy change type
            entropy_change_type = "increase" if delta_entropy > 0 else "decrease"
            
            # Record results
            result = {
                "node_id": target_node,
                "actual_class": actual_class_before,
                "predicted_class_before": predicted_class_before,
                "predicted_class_after": int(predicted_class_after),
                "probabilities_before": probs_before.tolist(),
                "probabilities_after": probs_after.tolist(),
                "entropy_before": float(entropy_before),
                "entropy_after": float(entropy_after),
                "delta_entropy": float(delta_entropy),
                "kl_divergence": float(kl_divergence),
                "attack_history": [int(x) for x in attack_history],
                "misclassified": misclassified.cpu().detach().numpy(),
                "entropy_change_type": entropy_change_type,
                "budget_used": budget,
                "experiment_timestamp": time.time()
            }
            
            logger.info(f"Node {target_node}: ΔH={delta_entropy:.4f}, KL={kl_divergence:.4f}, "
                       f"Misclassified={misclassified}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing node {target_node}: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_experiment(self, num_nodes=200, budget=5):
        """
        Run the full experiment on multiple nodes
        """
        logger.info(f"Starting Connectivity-Disruption Theory experiment")
        logger.info(f"Dataset: {self.dataset}, Surrogate: {self.surrogate_model}")
        logger.info(f"Target nodes: {num_nodes}, Budget: {budget}")
        
        # Select nodes for experiment
        selected_nodes = self.target_nodes[:num_nodes] if len(self.target_nodes) >= num_nodes else self.target_nodes
        
        successful_experiments = 0
        misclassified_count = 0
        entropy_increase_count = 0
        entropy_decrease_count = 0
        
        start_time = time.time()
        
        for i, target_node in enumerate(tqdm(selected_nodes, desc="Processing nodes")):
            result = self.run_single_node_experiment(target_node, budget)
            
            if result is not None:
                self.experiment_results.append(result)
                successful_experiments += 1
                
                if result["misclassified"]:
                    misclassified_count += 1
                
                if result["delta_entropy"] > 0:
                    entropy_increase_count += 1
                else:
                    entropy_decrease_count += 1
            
            # Save intermediate results every 50 nodes
            if (i + 1) % 50 == 0:
                self.save_results(f"intermediate_results_{i+1}.json")
        
        end_time = time.time()
        
        # Compute summary statistics
        summary = self.compute_summary_statistics()
        summary["total_time_seconds"] = end_time - start_time
        summary["successful_experiments"] = successful_experiments
        summary["misclassified_count"] = misclassified_count
        summary["entropy_increase_count"] = entropy_increase_count
        summary["entropy_decrease_count"] = entropy_decrease_count
        
        logger.info(f"Experiment completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Successfully processed {successful_experiments} nodes")
        logger.info(f"Misclassified: {misclassified_count}/{successful_experiments}")
        logger.info(f"Entropy increases: {entropy_increase_count}")
        logger.info(f"Entropy decreases: {entropy_decrease_count}")
        
        return summary
    
    def compute_summary_statistics(self):
        """
        Compute summary statistics for the experiment
        """
        if not self.experiment_results:
            return {}
        
        # Extract values
        delta_entropies = [r["delta_entropy"] for r in self.experiment_results]
        kl_divergences = [r["kl_divergence"] for r in self.experiment_results]
        misclassified = [r["misclassified"] for r in self.experiment_results]
        
        # Separate by entropy change type
        entropy_increases = [r for r in self.experiment_results if r["delta_entropy"] > 0]
        entropy_decreases = [r for r in self.experiment_results if r["delta_entropy"] < 0]
        
        summary = {
            "dataset": self.dataset,
            "surrogate_model": self.surrogate_model,
            "total_nodes_tested": len(self.experiment_results),
            "misclassification_rate": np.mean(misclassified),
            "entropy_statistics": {
                "mean_delta_entropy": np.mean(delta_entropies),
                "std_delta_entropy": np.std(delta_entropies),
                "min_delta_entropy": np.min(delta_entropies),
                "max_delta_entropy": np.max(delta_entropies)
            },
            "kl_divergence_statistics": {
                "mean_kl_divergence": np.mean(kl_divergences),
                "std_kl_divergence": np.std(kl_divergences),
                "min_kl_divergence": np.min(kl_divergences),
                "max_kl_divergence": np.max(kl_divergences)
            },
            "entropy_increase_analysis": {
                "count": len(entropy_increases),
                "misclassification_rate": np.mean([r["misclassified"] for r in entropy_increases]) if entropy_increases else 0,
                "mean_kl_divergence": np.mean([r["kl_divergence"] for r in entropy_increases]) if entropy_increases else 0
            },
            "entropy_decrease_analysis": {
                "count": len(entropy_decreases),
                "misclassification_rate": np.mean([r["misclassified"] for r in entropy_decreases]) if entropy_decreases else 0,
                "mean_kl_divergence": np.mean([r["kl_divergence"] for r in entropy_decreases]) if entropy_decreases else 0
            }
        }
        
        return summary
    
    def save_results(self, filename="connectivity_disruption_results.json"):
        """
        Save experiment results to JSON file
        """
        output_data = {
            "experiment_metadata": {
                "dataset": self.dataset,
                "surrogate_model": self.surrogate_model,
                "defense_model": self.defense_model,
                "timestamp": time.time()
            },
            "summary_statistics": self.compute_summary_statistics(),
            "detailed_results": self.experiment_results
        }
        print(output_data)
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
    
    def analyze_theorem_validation(self):
        """
        Analyze results to validate the Connectivity-Disruption Theory
        """
        if not self.experiment_results:
            logger.warning("No results to analyze")
            return
        
        # Count cases supporting the theorem
        theorem_support_cases = 0
        total_misclassified = 0
        
        for result in self.experiment_results:
            if result["misclassified"]:
                total_misclassified += 1
                
                # Check theorem conditions
                delta_h = result["delta_entropy"]
                kl_div = result["kl_divergence"]
                
                # Theorem is supported if:
                # 1. KL divergence increases (should be > 0)
                # 2. Either entropy increases (uncertainty) or decreases (false certainty)
                if kl_div > 0 and (delta_h > 0 or delta_h < 0):
                    theorem_support_cases += 1
        
        support_rate = theorem_support_cases / total_misclassified if total_misclassified > 0 else 0
        
        logger.info("=== Theorem Validation Analysis ===")
        logger.info(f"Total misclassified nodes: {total_misclassified}")
        logger.info(f"Cases supporting theorem: {theorem_support_cases}")
        logger.info(f"Theorem support rate: {support_rate:.2%}")
        
        return {
            "total_misclassified": total_misclassified,
            "theorem_support_cases": theorem_support_cases,
            "support_rate": support_rate
        }

def main():
    """
    Main function to run the Connectivity-Disruption Theory experiment
    """
    # Experiment configuration
    DATASET = 'cora'
    SURROGATE_MODEL = 'gcn'
    DEFENSE_MODEL = 'gcn'
    NUM_NODES = 200
    BUDGET = 5
    
    # Initialize experiment
    experiment = ConnectivityDisruptionExperiment(
        surrogate_model=SURROGATE_MODEL,
        dataset=DATASET,
        defense_model=DEFENSE_MODEL
    )
    
    # Run experiment
    logger.info("Starting Connectivity-Disruption Theory Experimental Proof")
    summary = experiment.run_experiment(num_nodes=NUM_NODES, budget=BUDGET)
    
    # Save results
    experiment.save_results("connectivity_disruption_experiment_results.json")
    
    # Analyze theorem validation
    validation_results = experiment.analyze_theorem_validation()
    
    # Print final summary
    print("\n" + "="*60)
    print("CONNECTIVITY-DISRUPTION THEORY EXPERIMENTAL PROOF")
    print("="*60)
    print(f"Dataset: {DATASET}")
    print(f"Nodes tested: {summary.get('successful_experiments', 0)}")
    print(f"Misclassification rate: {summary.get('misclassification_rate', 0):.2%}")
    print(f"Theorem support rate: {validation_results.get('support_rate', 0):.2%}")
    print(f"Mean ΔH: {summary['entropy_statistics']['mean_delta_entropy']:.4f}")
    print(f"Mean KL divergence: {summary['kl_divergence_statistics']['mean_kl_divergence']:.4f}")
    print("="*60)
    
    return experiment

if __name__ == "__main__":
    experiment = main()