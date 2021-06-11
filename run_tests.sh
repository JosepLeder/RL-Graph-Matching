python train_and_test.py --test True --agent "DFS"
python train_and_test.py --test True --agent "DDQN_DFS"
python train_and_test.py --tau 0.995 --test True --agent "DDQN" --batch_size 64 --sample_size 0 --replay_size 1000 --maxsteps=1000000
python train_and_test.py --tau 0.995 --test True --agent "pretrain_DDQN" --batch_size 64 --sample_size 0 --replay_size 1000 --maxsteps=1000000

