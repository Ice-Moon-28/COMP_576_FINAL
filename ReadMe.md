nohup python main.py --model meta-llama/Llama-2-7b-hf --dataset coqa --device cuda --num_generations_per_prompt 10 > output1.log 2>&1 &

python main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device cuda --num_generations_per_prompt 10
python knock_out_main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device mps --num_generations_per_prompt 10 --batch_size 1
python knock_out_main.py --model meta-llama/Llama-2-7b-hf  --dataset coqa --device cuda --num_generations_per_prompt 10 --batch_size 1


python main.py --model meta-llama/Llama-2-7b-hf --dataset squad --device cuda --num_generations_per_prompt 10

git config --global user.name "Linghua Zhang"
git config --global user.email "zlh20011228@gmail.com"