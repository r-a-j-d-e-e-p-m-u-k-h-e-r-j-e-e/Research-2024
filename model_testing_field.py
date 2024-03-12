import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import ast  # Import for safely evaluating strings containing Python literals

def main():
    model_directory = "/home/paperspace/rajdeep/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_opt350_success/"
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_directory, local_files_only=True)

    input_file = "/home/paperspace/rajdeep/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/accuracy_testing/eval.json"
    output_file = 'output_predictions_test_your_model_name.json'
    process_evaluation_dataset(input_file, output_file, tokenizer, model)
    
def generate_tcp_packet(prompt, tokenizer, model):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(input_ids, max_length=700, num_return_sequences=1)
        end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    time_taken = end_time - start_time

    # Attempt to extract just the predicted TCP packet part
    # Assuming it's the last dictionary in the generated text, followed by "</s>"
    try:
        # Splitting the generated text by "</s>" and taking the last part before "</s>"
        predicted_part = generated_text.rsplit("</s>", 1)[0].rsplit("{", 1)[-1]
        # Re-adding the curly braces that were removed by rsplit
        predicted_tcp_packet = "{" + predicted_part + "}"
    except Exception as e:
        print(f"Failed to extract predicted TCP packet: {e}")
        predicted_tcp_packet = "{}"  # Default to an empty dict representation if extraction fails

    return predicted_tcp_packet, time_taken


def process_evaluation_dataset(input_file, output_file, tokenizer, model):
    with open(input_file, 'r') as file:
        data = json.load(file)[:500]  # Limit the data to the first 500 items

    output_data = []
    field_accuracy_counts = {}  # Track correct counts per field
    field_total_counts = {}  # Track total counts per field

    for item in data:
        prompt = item['prompt']
        correct_answer_json = item['chosen']  # Assuming 'chosen' is in a JSON-like string format

        predicted_tcp_packet, time_taken = generate_tcp_packet(prompt, tokenizer, model)

        # Convert JSON strings to dictionaries for field-by-field comparison
        try:
            predicted_dict = ast.literal_eval(predicted_tcp_packet)
            correct_dict = ast.literal_eval(correct_answer_json)
            update_field_counts(predicted_dict, correct_dict, field_accuracy_counts, field_total_counts)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing predicted or correct answer into dictionary: {e}")

        output_data.append({
            'Prompt': prompt,
            'Predicted_TCP_Packet': predicted_tcp_packet,
            'Correct_Answer': correct_answer_json,
            'Time_Taken': time_taken
        })

        print("Processed one item. Time Taken:", time_taken, "seconds")

    # Calculate and print field-level accuracy for each field
    field_accuracies = {field: (field_accuracy_counts.get(field, 0) / field_total_counts.get(field, 1)) * 100
                        for field in field_total_counts}

    for field, accuracy in field_accuracies.items():
        print(f"Accuracy for {field}: {accuracy:.2f}%")

    # Save the predictions and field accuracies to the output file
    save_predictions(output_data, field_accuracies, output_file)

def update_field_counts(predicted_dict, correct_dict, field_accuracy_counts, field_total_counts):
    for field in correct_dict:
        field_total_counts[field] = field_total_counts.get(field, 0) + 1
        if predicted_dict.get(field) == correct_dict[field]:
            field_accuracy_counts[field] = field_accuracy_counts.get(field, 0) + 1

def save_predictions(output_data, field_accuracies, output_file):
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
        outfile.write("\nField-level accuracies:\n")
        json.dump(field_accuracies, outfile, indent=4)
    print(f"Predictions and field-level accuracies saved to: {output_file}")

if __name__ == "__main__":
    main()

