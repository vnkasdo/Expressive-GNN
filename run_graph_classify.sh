
# for simple graph
dataset=MUTAG
for fold in {0..9}
do 
    python graph_classification.py --dataset $dataset --hidden_dim 16 --phi MLP --device 0 --fold_idx $fold --lr 0.01 --agg cat  --first_phi   
done

# for attributed graph
dataset=Synthie
for fold in {0..9}
do 
    python graph_classification.py --dataset $dataset --hidden_dim 16 --phi MLP --device 0 --fold_idx $fold --lr 0.01 --agg cat --attribute --first_phi   
done


