python main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device cuda --num_generations_per_prompt 10
python knock_out_main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device mps --num_generations_per_prompt 10 --batch_size 1
python knock_out_main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device cuda --num_generations_per_prompt 10 --batch_size 1