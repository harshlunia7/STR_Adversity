echo "evaluate model on rain restored by promptir"
python evaluate_abinet_on_six_benchmarks.py \
--config configs/train_abinet_non_degraded.yaml \
--checkpoint train_abinet_nondegraded/train-abinet/train-abinet_9_16000.pth \
--gt_file ../eval_datasets_str_english/real/image_restored_real_str_adverse/promptir/combined/rain_gt_real_restored.txt \
--input ../eval_datasets_str_english/real/image_restored_real_str_adverse/promptir;

echo "evaluate model on rain restored by wgws_net"
python evaluate_abinet_on_six_benchmarks.py \
--config configs/train_abinet_non_degraded.yaml \
--checkpoint train_abinet_nondegraded/train-abinet/train-abinet_9_16000.pth \
--gt_file ../eval_datasets_str_english/real/image_restored_real_str_adverse/wgws_net/combined/rain_gt_real_restored.txt \
--input ../eval_datasets_str_english/real/image_restored_real_str_adverse/wgws_net;

echo "evaluate model on rain not restored"
python evaluate_abinet_on_six_benchmarks.py \
--config configs/train_abinet_non_degraded.yaml \
--checkpoint train_abinet_nondegraded/train-abinet/train-abinet_9_16000.pth \
--gt_file ../eval_datasets_str_english/real/checking_for_not_image_restored/short_real_str/rain_gt_real_restored.txt \
--input ../eval_datasets_str_english/real/checking_for_not_image_restored/;