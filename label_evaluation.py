from vllm import LLM, SamplingParams
from label_retriever import LabelRetriever
from configs.label_config import LabelConfig
import numpy as np
import os
import argparse
from pathlib import Path
import yaml
import shelve
from scipy.stats import binomtest
import math
import random
import time

def load_config(config_path):
    label_eval_dir = Path(__file__).resolve().parent
    config_path = label_eval_dir / "configs" / config_path
    return LabelConfig.from_yaml(str(config_path))


class LabelEvaluation:
    def __init__(self, config: LabelConfig):
        self.config = config
        self.weight_network = LabelRetriever(config) 
        self.n_sampled_features = self.weight_network.sample_indices.shape[1]

        self.max_layer = config.n_layers - 1
        self.min_layer = 0
        self.save_scores_path = str(config.eval_dir / "scores")
        self.m = config.m
        self.n_questions = config.n_questions
        self.character_1 = config.character_1
        self.character_2 = config.character_2

        self.llm = None
        self.tokenizer = None
        self.current_model_id = config.judge_model_absolute
        self.token_ids_1 = []
        self.token_ids_2 = []
        assert self.m <= .5, "m should be less than or equal to .5 to ensure there is a difference in distribution between top and bottom neighbors."

    def _lazy_load_judge_model(self, judge_model_id):
        if self.llm is None or self.current_model_id != judge_model_id:
            print(f"Loading judge model {judge_model_id}...")
            self.llm = LLM(model = judge_model_id, enable_prefix_caching = True, gpu_memory_utilization = .7)
            self.tokenizer = self.llm.get_tokenizer()
            self.current_model_id = judge_model_id
            target_strings_1 = [self.character_1, f" {self.character_1}"]
            target_strings_2 = [self.character_2, f" {self.character_2}"]
            if self.character_1.isalpha():
                if self.character_1.isupper():
                    target_strings_1 += [self.character_1.lower(), f" {self.character_1.lower()}"]
                else:
                    target_strings_1 += [self.character_1.upper(), f" {self.character_1.upper()}"]
            if self.character_2.isalpha():
                if self.character_2.isupper():
                    target_strings_2 += [self.character_2.lower(), f" {self.character_2.lower()}"]
                else:
                    target_strings_2 += [self.character_2.upper(), f" {self.character_2.upper()}"]

            self.token_ids_1 = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in target_strings_1]
            self.token_ids_2 = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in target_strings_2]
        


    def show_evaluation_results(self):
        if not os.path.exists(self.save_scores_path + ".db"):
            print("No evaluation results found.")
            return
        with shelve.open(self.save_scores_path) as db:
            for key in db:
                print(f"{key}: {db[key]}")
    def save_evaluation_result(self, layer:int, feature_idx:int, downstream:bool, score:float):
        assert isinstance(downstream, bool), f"downstream should be a boolean value instead of {type(downstream)}: {downstream}"
        self.config.lock_parameters()

        type_key = "downstream" if downstream else "upstream"
        key = f"{layer}_{feature_idx}_{type_key}"
        with shelve.open(self.save_scores_path, "c") as db:
            if key not in db:
                db[key] = {"score": score, "n_questions": self.n_questions}
            else:
                temp_dict = db[key]
                temp_dict["score"] = score
                temp_dict["n_questions"] = self.n_questions
                db[key] = temp_dict
    def get_evaluation_result(self, layer, feature_idx, downstream):
        assert isinstance(downstream, bool), f"downstream should be a boolean value instead of {type(downstream)}: {downstream}"
        if not os.path.exists(self.save_scores_path + ".db"):
            return {}
        type_key = "downstream" if downstream else "upstream"
        key = f"{layer}_{feature_idx}_{type_key}"
        with shelve.open(self.save_scores_path) as db:
            if key in db:
                return db[key]
            else:
                return {}
    

    def sample_features_to_evaluate(self, layer, num_features):
        """Returns a numpy array of random feature indices to evaluate for a given layer. Assumes sample_indices_path is provided."""
        assert (self.weight_network.sample_indices is not None), "Sample indices not loaded. Please provide a valid sample_indices_path."
        sampled_indices = self.weight_network.sample_indices[layer]
        selected_indices = np.random.choice(sampled_indices, size=num_features, replace=False)
        return selected_indices.tolist()
    
    def get_neighbor_quiz(self, layer, feature_idx, downstream = True):
        if (layer == self.max_layer) and downstream:
            raise ValueError(f"Layer {self.max_layer} is the output layer and has no downstream neighbors.")
        if (layer == self.min_layer) and not downstream:
            raise ValueError(f"Layer {self.min_layer} is the input layer and has no upstream neighbors.")

        
    
        #FIXME - taking ALL of the downstream features and then sampling uniformly from the top percent of them creates bias because the number of downstream features varies by layer.
        n_downstream_features = self.n_sampled_features * (self.max_layer - layer) if downstream else self.n_sampled_features * (layer - self.min_layer)
        k = int(n_downstream_features * self.m)
        assert k > self.n_questions, f"m is too small to generate {self.n_questions} questions. k = {k}, n_questions = {self.n_questions}"
        neighbor_func = self.weight_network.get_k_downstream_neighbors if downstream else self.weight_network.get_k_upstream_neighbors
        kwargs = {"layer": layer, "feature_idx": feature_idx, "k": k,"index_in_sampled": True}
        if downstream: kwargs["max_layer"] = self.max_layer
        if not downstream: kwargs["min_layer"] = self.min_layer
        

        # I originally code this downstream-specific but later adapted it to work for upstream as well. The variable names have not changed yet
        k_downstream_top = neighbor_func(**kwargs, method = "top")
        k_downstream_bottom = neighbor_func(**kwargs, method = "abs_bottom")

        # I should sample more than n_questions so that I can drop the neighbors that have nan descriptions. int(1.5 *n_questions) should do the trick
        to_sample = int(3 * self.n_questions)
        if len(k_downstream_top) < to_sample or len(k_downstream_bottom) < to_sample:
            raise ValueError(f"Tried to sample {to_sample} neighbors, but only found {len(k_downstream_top)} top neighbors and {len(k_downstream_bottom)} bottom neighbors.")
        k_downstream_top_sampled = np.array(k_downstream_top)[np.random.choice(len(k_downstream_top), size=to_sample, replace=False), :]
        k_downstream_bottom_sampled = np.array(k_downstream_bottom)[np.random.choice(len(k_downstream_bottom), size=to_sample, replace=False), :]

        # Get labels for sampled neighbors
        top_neighbors, source = self.weight_network.get_labels_for_neighbors(layer = layer, feature_idx = feature_idx, neighbor_results = k_downstream_top_sampled, index_in_sampled = True, additional_label_info=['typeName'], show_source_feature = False, show_neighbors=False)
        bottom_neighbors, _ = self.weight_network.get_labels_for_neighbors(layer = layer, feature_idx = feature_idx, neighbor_results = k_downstream_bottom_sampled, index_in_sampled = True, additional_label_info=['typeName'], show_source_feature = False, show_neighbors = False)
        top_neighbors.dropna(subset = ["description"], inplace = True)
        bottom_neighbors.dropna(subset = ["description"], inplace = True)

        # Create a priority score column that helps us know which labels to keep if there are duplicates
        priority_map = {"np_max-act": 1, "np_max-act-logits": 2}
        top_neighbors["priority"] = top_neighbors["typeName"].map(priority_map).fillna(3)
        bottom_neighbors["priority"] = bottom_neighbors["typeName"].map(priority_map).fillna(3)
        source["priority"] = source["typeName"].map(priority_map).fillna(3)
        top_neighbors_sorted = top_neighbors.sort_values(by=["original_feature_idx", "layer", "priority"])
        bottom_neighbors_sorted = bottom_neighbors.sort_values(by=["original_feature_idx", "layer", "priority"])
        source_sorted = source.sort_values(by=["original_feature_idx", "layer", "priority"])
        top_neighbors_deduped = top_neighbors_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")
        bottom_neighbors_deduped = bottom_neighbors_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")
        source_deduped = source_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")

        # Only keep n_questions
        quiz_len = min(len(top_neighbors_deduped), len(bottom_neighbors_deduped), self.n_questions)
        if quiz_len < self.n_questions:
            print(f"Warning: Did not find {self.n_questions} questions with valid descriptions. Did find {quiz_len} valid questions.")
        top_neighbors = top_neighbors_deduped.sample(n=quiz_len)
        bottom_neighbors = bottom_neighbors_deduped.sample(n=quiz_len)

        if len(source_deduped) == 0:
            raise ValueError("Source feature has no valid description. Cannot generate quiz.")
        source_description = source_deduped["description"].iloc[0] 
        top_descriptions = top_neighbors["description"].tolist()
        bottom_descriptions = bottom_neighbors["description"].tolist()

        return source_description, top_descriptions, bottom_descriptions
    
    def evaluate_with_judge(self, layer, feature_idx, judge_model_id = None, downstream = True, save_results = True):
        start = time.time()
        if judge_model_id is None:
            judge_model_id = self.current_model_id
        
        # Get stuff to label
        source_description, top_descriptions, bottom_descriptions = self.get_neighbor_quiz(layer, feature_idx, downstream = downstream)
        if len(top_descriptions) < self.n_questions or len(bottom_descriptions) < self.n_questions:
            raise ValueError(f"Not enough valid descriptions to generate quiz. Found {len(top_descriptions)} top descriptions and {len(bottom_descriptions)} bottom descriptions, but need at least {self.n_questions} of each.")
        correct_answers = np.random.choice([self.character_1, self.character_2], size=len(top_descriptions))

        # Load the model
        self._lazy_load_judge_model(judge_model_id)

        # Set up prompt
        preamble = "You are an evaluator of feature labels within a neural network."
        if downstream:
            description = "You are given a source feature description and two downstream feature descriptions. One of these downstream features is strongly influenced by the source feature, while the other is weakly connected. Your task is to determine which downstream feature is strongly connected to the source feature based on the descriptions. Descriptions of related concepts are more likely to indicate a strong connection between features."
        else:
            description = "You are given a target feature description and two upstream feature descriptions. One of these upstream features has a strong causal influence over the target feature, while the other has a weak connection. Your task is to determine which upstream feature is more likely to have a strong influence on the target feature based on the descriptions. Descriptions of related concepts are more likely to indicate a strong connection between features."
        formatting_instructions = f"Respond with exactly one character: '{self.character_1}' or '{self.character_2}', corresponding to the better option. Do not include any punctuation, formatting, or explanations."
        source_or_target = "Source" if downstream else "Target"
        upstream_or_downstream = "downstream" if downstream else "upstream"
        few_shot = f"Example:\n{source_or_target} Feature: Texas\n{self.character_1}: Dallas\n{self.character_2}: Oxygen\nAnswer: {self.character_1}\n\nExample:\n{source_or_target} Feature: Happiness\n{self.character_1}: grass\n{self.character_2}: Elation\nAnswer: {self.character_2}"

        system_prompt = f"{preamble} {description} {formatting_instructions}\n\n{few_shot}"
        question_text = f"Which {upstream_or_downstream} feature description is more likely to have a strong connection to the {source_or_target.lower()} feature description? ({self.character_1} or {self.character_2})"
        # Build prompt in batched format
        prompts = []
        for top_desc, bottom_desc, correct_position in zip(top_descriptions, bottom_descriptions, correct_answers):

            # Build the prompt
            assert correct_position in [self.character_1, self.character_2], f"Correct position should be '{self.character_1}' or '{self.character_2}', but got {correct_position}"
            if correct_position == self.character_1:
                position_1 = top_desc
                position_2 = bottom_desc
            else:
                position_1 = bottom_desc
                position_2 = top_desc

            user_prompt = f"{question_text}\n\n{source_or_target} Feature: {source_description}\n{self.character_1}: {position_1}\n{self.character_2}: {position_2}"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
            prompts.append(text)

        # Get the result of the entire batch
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs = 20)
        outputs = self.llm.generate(prompts, sampling_params = sampling_params)
        
        # Process the results
        probs_1_list = []
        probs_2_list = []
        for i, output in enumerate(outputs):
            first_token_logprobs = output.outputs[0].logprobs[0]
        
            prob_1 = 0.0
            prob_2 = 0.0

            # Add probabilities for all tokens corresponding to "1" and "2"
            for tid in self.token_ids_1:
                if tid in first_token_logprobs:
                    prob_1 += math.exp(first_token_logprobs[tid].logprob)       
            for tid in self.token_ids_2:
                if tid in first_token_logprobs:
                    prob_2 += math.exp(first_token_logprobs[tid].logprob)
            probs_1_list.append(prob_1)
            probs_2_list.append(prob_2)

        assert len(probs_1_list) == len(correct_answers) == len(probs_2_list), "Length of probabilities and correct answers should be the same."
        probs_1_array = np.array(probs_1_list)
        probs_2_array = np.array(probs_2_list)
        model_choices = np.where(probs_2_array > probs_1_array, self.character_2, self.character_1)
        accuracy = np.mean(model_choices == correct_answers)
        print(f"Num {self.character_1} choices:", np.sum(probs_1_array >= probs_2_array), f"Num {self.character_2} choices:", np.sum(probs_2_array > probs_1_array), f"Score: {accuracy:.2f}")

        if save_results:
            self.save_evaluation_result(layer = layer, feature_idx = feature_idx, downstream = downstream, score = accuracy)

        end = time.time()
        print(f"Evaluation completed in {end - start:.2f} seconds")
        return accuracy, correct_answers, probs_1_array, probs_2_array

    def get_system_question_prompt(self, downstream = True):
        with open(self.config.label_eval_dir / "judge_prompts" / self.config.prompt_file, "r") as f:
            prompt_dict = yaml.safe_load(f)
        
        preamble = prompt_dict["preamble"]
        downstream_description = prompt_dict["downstream_description"]
        upstream_description = prompt_dict["upstream_description"]
        formatting_instructions = prompt_dict["formatting_instructions"].format(self.character_1, self.character_2)
        source_or_target = "Source" if downstream else "Target"
        upstream_or_downstream = "downstream" if downstream else "upstream"
        few_shot = prompt_dict["few_shot"].format(source_or_target, self.character_1, self.character_2, self.character_1, source_or_target, self.character_1, self.character_2, self.character_2)
        question_text = prompt_dict["question_text"].format(upstream_or_downstream, source_or_target.lower(), self.character_1, self.character_2)
        system_prompt = f"{preamble} {downstream_description if downstream else upstream_description} {formatting_instructions}\n\n{few_shot}"
        return system_prompt, question_text

    def evaluate_batch_with_judge(self, feature_tuples, judge_model_id=None, downstream=True, save_results=True):
            """
            Evaluates a batch of (layer, feature_idx) tuples in a single vLLM generation call.
            """
            start = time.time()
            if judge_model_id is None:
                judge_model_id = self.current_model_id
                
            # 1. Load the model and tokenizer early so we can build prompts
            self._lazy_load_judge_model(judge_model_id)


            # Set up prompt
            system_prompt, question_text = self.get_system_question_prompt(downstream = downstream)
            source_or_target = "Source" if downstream else "Target"


            all_prompts = []
            tracking_info = [] # Keeps track of which prompt belongs to which feature
            MAX_CHAR_LEN = 1500 # Truncation limit to prevent OOM spikes

            print(f"Assembling prompts for {len(feature_tuples)} features...")
            
            # --- PHASE 1: PROMPT ASSEMBLY ---
            for layer, feature_idx in feature_tuples:
                # Skip if already evaluated
                if save_results and self.get_evaluation_result(layer, feature_idx, downstream) != {}:
                    continue
                    
                try:
                    source_description, top_descriptions, bottom_descriptions = self.get_neighbor_quiz(layer, feature_idx, downstream=downstream)
                except Exception as e:
                    print(f"Skipping Layer {layer}, Feature {feature_idx}: {e}")
                    continue

                # Defensive truncation
                source_description = source_description[:MAX_CHAR_LEN]
                top_descriptions = [desc[:MAX_CHAR_LEN] for desc in top_descriptions]
                bottom_descriptions = [desc[:MAX_CHAR_LEN] for desc in bottom_descriptions]

                if len(top_descriptions) < self.n_questions or len(bottom_descriptions) < self.n_questions:
                    print(f"Skipping Layer {layer}, Feature {feature_idx}: Not enough valid descriptions.")
                    continue

                correct_answers = np.random.choice([self.character_1, self.character_2], size=len(top_descriptions))

                for top_desc, bottom_desc, correct_position in zip(top_descriptions, bottom_descriptions, correct_answers):
                    if correct_position == self.character_1:
                        position_1 = top_desc
                        position_2 = bottom_desc
                    else:
                        position_1 = bottom_desc
                        position_2 = top_desc

                    user_prompt = f"{question_text}\n\n{source_or_target} Feature: {source_description}\n{self.character_1}: {position_1}\n{self.character_2}: {position_2}"
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    all_prompts.append(text)
                    tracking_info.append({
                        "layer": layer,
                        "feature_idx": feature_idx,
                        "correct_answer": correct_position
                    })

            if not all_prompts:
                print("No valid prompts generated. Exiting batch evaluation.")
                return

            print(f"Successfully built {len(all_prompts)} total prompts. Starting vLLM inference...")

            # --- PHASE 2: UNIFIED INFERENCE ---
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
            outputs = self.llm.generate(all_prompts, sampling_params=sampling_params)

            # --- PHASE 3: SCORE UNPACKING ---
            results_by_feature = {}
            
            for tracking, output in zip(tracking_info, outputs):
                key = (tracking["layer"], tracking["feature_idx"])
                if key not in results_by_feature:
                    results_by_feature[key] = {"correct_answers": [], "probs_1": [], "probs_2": []}

                first_token_logprobs = output.outputs[0].logprobs[0]
                prob_1, prob_2 = 0.0, 0.0

                for tid in self.token_ids_1:
                    if tid in first_token_logprobs: prob_1 += math.exp(first_token_logprobs[tid].logprob)
                for tid in self.token_ids_2:
                    if tid in first_token_logprobs: prob_2 += math.exp(first_token_logprobs[tid].logprob)

                results_by_feature[key]["correct_answers"].append(tracking["correct_answer"])
                results_by_feature[key]["probs_1"].append(prob_1)
                results_by_feature[key]["probs_2"].append(prob_2)

            # Calculate final accuracy and save
            for (layer, feature_idx), data in results_by_feature.items():
                correct_answers = np.array(data["correct_answers"])
                probs_1_array = np.array(data["probs_1"])
                probs_2_array = np.array(data["probs_2"])

                model_choices = np.where(probs_2_array > probs_1_array, self.character_2, self.character_1)
                accuracy = np.mean(model_choices == correct_answers)

                if save_results:
                    self.save_evaluation_result(layer=layer, feature_idx=feature_idx, downstream=downstream, score=accuracy)

            end = time.time()
            print(f"Batch evaluation completed in {end - start:.2f} seconds.")
    
    def display_quiz(self, source_description, top_descriptions, bottom_descriptions, downstream = True):
        quiz_len = len(top_descriptions)
        assert quiz_len == len(bottom_descriptions), "Number of top and bottom descriptions should be the same."
        print(f"Source feature description: {source_description}\n")
        correct_answers = np.random.choice([self.character_1, self.character_2], size=quiz_len)

        options = {self.character_1: f"choose answer {self.character_1}",
                   self.character_2: f"choose answer {self.character_2}",
                   'x': "exit quiz",
                   "r": "go back one question"
                   }

        answer_choices = []

        i=0
        source_or_target = "Source" if downstream else "Target"
        upstream_or_downstream = "downstream" if downstream else "upstream"
        while i < quiz_len:
            print(f"Question {i+1}:")
            print("Source feature description:", source_description)
            if correct_answers[i] == self.character_1:
                print(f"{self.character_1})", top_descriptions[i])
                print(f"{self.character_2})", bottom_descriptions[i])
            else:
                print(f"{self.character_1})", bottom_descriptions[i])
                print(f"{self.character_2})", top_descriptions[i])

            input_str = f"Which {upstream_or_downstream} neighbor is more likely to be influenced by the {source_or_target.lower()} feature? ({self.character_1}/{self.character_2}), or enter 'x' to exit, 'r' to go back: " if downstream else f"Which {upstream_or_downstream} neighbor is more likely to influence the {source_or_target.lower()} feature? ({self.character_1}/{self.character_2}), or enter 'x' to exit, 'r' to go back: "
            choice = input(input_str)
            while choice not in options.keys():
                choice = input(f"Invalid choice. Please enter '{self.character_1}' or '{self.character_2}': ")
            if choice == "x":
                print("Exiting quiz.")
                break
            elif choice == "r":
                if i > 0:
                    i -= 1
                    answer_choices.pop()
                else:
                    print("Already at the first question. Cannot go back further.")
            else:
                answer_choices.append(choice)
                i += 1
        
        return answer_choices, correct_answers
    
    def _score_quiz(self, user_choices, correct_answers):
        num_responses = max(min(len(user_choices), len(correct_answers)), 1)
        
        score = sum([1 for user_choice, correct_answer in zip(user_choices, correct_answers) if user_choice == correct_answer])/num_responses
        return score
    
    def take_quiz_for_feature(self, layer, feature_idx, downstream = True, save_results = True):
        # Prevent accidental overwriting
        if save_results:
            existing_result = self.get_evaluation_result(layer, feature_idx, downstream)
            if existing_result != {}:
                response = input(f"Existing evaluation result found: {existing_result}. Do you want to overwrite it? (y/n): ")
                while response not in ["y", "n"]:
                    response = input("Invalid response. Please enter 'y' to overwrite or 'n' to keep existing result: ")
                if response == "n":
                    print("Keeping existing result. Exiting quiz.")
                    return False, None
                else:
                    print("Overwriting existing result. Starting quiz.")

        # Do the quiz
        source_description, top_descriptions, bottom_descriptions = self.get_neighbor_quiz(layer = layer, feature_idx = feature_idx, downstream = downstream)
        user_choices, correct_answers = self.display_quiz(source_description, top_descriptions, bottom_descriptions, downstream = downstream)
        quiz_complete = (len(user_choices) == len(correct_answers))
        score = self._score_quiz(user_choices, correct_answers)

        # Save results
        if save_results:
            if quiz_complete:
                self.save_evaluation_result(layer = layer, feature_idx = feature_idx, downstream = downstream, score = score)
            else:
                print("Quiz not completed. Results will not be saved.")


        print(f"Quiz complete: {quiz_complete}, Your score: {score:.2f}")
        return quiz_complete, score

    def hand_label_random_features(self, n_features_to_label, save_results = True):
        failed_attempts = 0
        features_labeled = 0
        while True:
            if failed_attempts >= 15:
                print("Too many failed attempts to find features with valid descriptions. Stopping.")
                break
            if features_labeled >= n_features_to_label:
                print(f"Labeled {features_labeled} features. Stopping.")
                break
            layer = np.random.randint(self.min_layer, self.max_layer + 1)
            feature_idx = np.random.randint(0, self.n_sampled_features)
            if layer == self.max_layer:
                downstream = False
            elif layer == self.min_layer:
                downstream = True
            else:
                downstream = bool(np.random.choice([True, False]))
            if self.get_evaluation_result(layer, feature_idx, downstream) != {}:
                failed_attempts += 1
                continue
            try:
                quiz_complete, score = self.take_quiz_for_feature(layer, feature_idx, downstream = downstream, save_results = save_results)
            except ValueError as e:
                quiz_complete = False
                failed_attempts += 1
                continue
            if quiz_complete:
                failed_attempts = 0
                features_labeled += 1
            else:
                failed_attempts += 1

            exit = input("Press 'x' to stop labeling, or any other key to continue: ")
            if exit == "x":
                break

    def get_label_stats(self, downstream = None):
        """Computes statistics for the saved feature evaluations.
        
        Args:
            downstream (bool or None): If True, only compute stats for downstream evaluations. If False, only compute stats for upstream evaluations. If None, compute stats for all evaluations.
        """
        if not os.path.exists(self.save_scores_path + ".db"):
            print("No evaluation results found.")
            return [], [], []
        
        with shelve.open(self.save_scores_path) as db:
            scores = []
            keys = []
            questions = []
            total_score = 0
            total_questions = 0
            for key in db:
                if downstream is not None:
                    if downstream and "downstream" not in key:
                        continue
                    if not downstream and "upstream" not in key:
                        continue
                result = db[key]
                if "score" in result and "n_questions" in result:
                    scores.append(result["score"])
                    keys.append(key)
                    questions.append(result["n_questions"])
            average_score = (np.array(scores) * np.array(questions)).sum() / sum(questions)
            print(f"Average score: {average_score}, Total questions answered: {sum(questions)}, Number of evaluations: {len(scores)}")
        return scores, questions, keys
    
    def calc_p_value(self, downstream = None):
        """Calculates a p-value for the observed average score compared to random guessing (0.5) using a binomial test."""
        
        scores, questions, keys = self.get_label_stats(downstream=downstream)
        if not scores:
            print("No evaluation results found.")
            return None
        total_correct = int(sum([score * n for score, n in zip(scores, questions)]))
        total_questions = int(sum(questions))
        p_value = binomtest(total_correct, total_questions, p=0.5, alternative = "greater").pvalue
        print(f"Total correct: {total_correct}, Total questions: {total_questions}, p-value: {p_value:.4f}")
        return p_value

            

        

def main(config):
    label_evaluator = LabelEvaluation(config)
    layers = np.arange(config.n_layers - 1)
    n_samples_per_layer = np.load(str(Path(config.sample_network_absolute) / "sampled_features.npy")).shape[1]
    indices = np.arange(n_samples_per_layer)
    combos = [(i, j) for i in layers for j in indices]
    random.seed(27)
    random.shuffle(combos)

    label_evaluator.evaluate_batch_with_judge(combos[400:600], downstream = True, save_results = True)
    label_evaluator.calc_p_value(downstream = True)

    print("\nShutting down vLLM gracefully...")
    if hasattr(label_evaluator, 'llm') and label_evaluator.llm is not None:
        del label_evaluator.llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("Run complete!")


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate feature labels based on feature-connections")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file relative to configs directory")
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)


        


