#!/bin/bash

################################################################################
# 涓€娆℃€ф潈闄愯缃剼鏈?################################################################################
# 杩欎釜鑴氭湰浼氳嚜鍔ㄨ缃墍鏈夐渶瑕佹墽琛屾潈闄愮殑鏂囦欢
# 杩愯鏂瑰紡锛歜ash fix_permissions.sh
################################################################################

echo "姝ｅ湪璁剧疆鏂囦欢鎵ц鏉冮檺..."

# 褰撳墠瀛樺湪鐨勮剼鏈?chmod +x verify_deployment.sh 2>/dev/null && echo "鉁?verify_deployment.sh" || echo "鈿?verify_deployment.sh 涓嶅瓨鍦?
chmod +x setup_hpc.sh 2>/dev/null && echo "鉁?setup_hpc.sh" || echo "鈿?setup_hpc.sh 涓嶅瓨鍦?
chmod +x start.sh 2>/dev/null && echo "鉁?start.sh" || echo "鈿?start.sh 涓嶅瓨鍦?
chmod +x run_reproduce.sh 2>/dev/null && echo "鉁?run_reproduce.sh" || echo "鈿?run_reproduce.sh 涓嶅瓨鍦?
chmod +x hpc_run_all.sh 2>/dev/null && echo "鉁?hpc_run_all.sh" || echo "鈿?hpc_run_all.sh 涓嶅瓨鍦?
chmod +x fix_permissions.sh 2>/dev/null && echo "鉁?fix_permissions.sh" || echo "鈿?fix_permissions.sh 涓嶅瓨鍦?

# SLURM 鑴氭湰
chmod +x submit_multi_seed.slurm 2>/dev/null && echo "鉁?submit_multi_seed.slurm" || echo "鈿?submit_multi_seed.slurm 涓嶅瓨鍦?

# 鑷姩鐢熸垚鐨勮剼鏈紙濡傛灉瀛樺湪锛?chmod +x monitor_jobs.sh 2>/dev/null && echo "鉁?monitor_jobs.sh" || echo "鈿?monitor_jobs.sh (杩愯 start.sh 鍚庝細鑷姩鐢熸垚)"
chmod +x collect_results.sh 2>/dev/null && echo "鉁?collect_results.sh" || echo "鈿?collect_results.sh (杩愯 start.sh 鍚庝細鑷姩鐢熸垚)"

echo ""
echo "鏉冮檺璁剧疆瀹屾垚锛?
echo ""
echo "娉ㄦ剰锛?
echo "  - monitor_jobs.sh 鍜?collect_results.sh 浼氬湪杩愯 start.sh 鍚庤嚜鍔ㄧ敓鎴?
echo "  - 鐢熸垚鍚庡畠浠細鑷姩鍏锋湁鎵ц鏉冮檺"
echo ""
echo "鐜板湪鍙互杩愯锛?
echo "  bash verify_deployment.sh"
