import time
import os
import sys
import json
import subprocess
from datetime import datetime

# 确保项目根目录在sys.path中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_script_with_timing(script_path, test_indices=None, max_time=3600):
    """运行脚本并记录时间"""
    print(f"开始运行脚本: {script_path}")
    start_time = time.time()
    
    try:
        # 如果指定了测试索引，修改脚本中的处理范围
        if test_indices:
            print(f"测试索引范围: {test_indices}")
        
        # 运行脚本
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        
        # 等待完成或超时
        try:
            stdout, stderr = process.communicate(timeout=max_time)
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                'success': process.returncode == 0,
                'execution_time': execution_time,
                'stdout': stdout,
                'stderr': stderr,
                'return_code': process.returncode
            }
        except subprocess.TimeoutExpired:
            process.kill()
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                'success': False,
                'execution_time': execution_time,
                'stdout': '',
                'stderr': '脚本执行超时',
                'return_code': -1
            }
            
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'success': False,
            'execution_time': execution_time,
            'stdout': '',
            'stderr': str(e),
            'return_code': -1
        }

def count_processed_files(output_dir):
    """统计处理完成的文件数量"""
    if not os.path.exists(output_dir):
        return 0
    
    count = 0
    for file in os.listdir(output_dir):
        if file.endswith('_ADMeanFused.tif') or file.endswith('_ADMeanFused_WithTiles.tif'):
            count += 1
    
    return count

def get_file_sizes(output_dir):
    """获取输出文件的总大小"""
    if not os.path.exists(output_dir):
        return 0
    
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    
    return total_size

def create_test_version(original_script, test_script, test_range):
    """创建测试版本的脚本，限制处理范围"""
    with open(original_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改处理范围
    if 'range(total_lakes)' in content:
        content = content.replace(
            'range(total_lakes)', 
            f'range({test_range[0]}, min({test_range[1]}, total_lakes))'
        )
    elif 'all_indices = list(range(total_lakes))' in content:
        content = content.replace(
            'all_indices = list(range(total_lakes))',
            f'all_indices = list(range({test_range[0]}, min({test_range[1]}, total_lakes)))'
        )
    
    with open(test_script, 'w', encoding='utf-8') as f:
        f.write(content)

def performance_test():
    """性能测试主函数"""
    print("=" * 60)
    print("数据下载性能对比测试")
    print("=" * 60)
    
    # 脚本路径
    original_script = r'd:\09_Code\Gis_Script\05_SAM\S1_GLDownload.py'
    optimized_script = r'd:\09_Code\Gis_Script\05_SAM\S1_GLDownload_Optimized.py'
    
    # 测试配置
    test_range = (0, 10)  # 测试前10个冰湖
    max_test_time = 1800  # 最大测试时间30分钟
    
    # 输出目录
    base_output_dir = r'E:\Dataset_and_Demo\SETP_GL'
    
    results = {}
    
    # 检查脚本是否存在
    if not os.path.exists(original_script):
        print(f"原始脚本不存在: {original_script}")
        return
    
    if not os.path.exists(optimized_script):
        print(f"优化脚本不存在: {optimized_script}")
        return
    
    # 创建测试版本
    original_test_script = original_script.replace('.py', '_test.py')
    optimized_test_script = optimized_script.replace('.py', '_test.py')
    
    try:
        create_test_version(original_script, original_test_script, test_range)
        create_test_version(optimized_script, optimized_test_script, test_range)
        
        print(f"测试范围: 索引 {test_range[0]} 到 {test_range[1]-1}")
        print(f"最大测试时间: {max_test_time} 秒")
        print()
        
        # 测试原始版本
        print("1. 测试原始版本...")
        original_result = run_script_with_timing(original_test_script, test_range, max_test_time)
        results['original'] = original_result
        
        print(f"原始版本执行时间: {original_result['execution_time']:.2f} 秒")
        print(f"原始版本执行状态: {'成功' if original_result['success'] else '失败'}")
        if not original_result['success']:
            print(f"错误信息: {original_result['stderr']}")
        print()
        
        # 等待一段时间再测试优化版本
        print("等待5秒后测试优化版本...")
        time.sleep(5)
        
        # 测试优化版本
        print("2. 测试优化版本...")
        optimized_result = run_script_with_timing(optimized_test_script, test_range, max_test_time)
        results['optimized'] = optimized_result
        
        print(f"优化版本执行时间: {optimized_result['execution_time']:.2f} 秒")
        print(f"优化版本执行状态: {'成功' if optimized_result['success'] else '失败'}")
        if not optimized_result['success']:
            print(f"错误信息: {optimized_result['stderr']}")
        print()
        
        # 性能对比分析
        print("=" * 60)
        print("性能对比结果")
        print("=" * 60)
        
        if original_result['success'] and optimized_result['success']:
            time_improvement = original_result['execution_time'] - optimized_result['execution_time']
            improvement_percentage = (time_improvement / original_result['execution_time']) * 100
            
            print(f"原始版本执行时间: {original_result['execution_time']:.2f} 秒")
            print(f"优化版本执行时间: {optimized_result['execution_time']:.2f} 秒")
            print(f"时间节省: {time_improvement:.2f} 秒")
            print(f"性能提升: {improvement_percentage:.1f}%")
            
            if improvement_percentage > 0:
                print("✅ 优化版本性能更好！")
            elif improvement_percentage < -10:
                print("❌ 优化版本性能较差")
            else:
                print("⚖️ 性能差异不明显")
        else:
            print("⚠️ 无法进行完整的性能对比，因为有脚本执行失败")
        
        # 统计输出文件
        print("\n文件输出统计:")
        for subdir in os.listdir(base_output_dir):
            subdir_path = os.path.join(base_output_dir, subdir)
            if os.path.isdir(subdir_path):
                file_count = count_processed_files(subdir_path)
                total_size = get_file_sizes(subdir_path)
                print(f"  {subdir}: {file_count} 个文件, {total_size / (1024*1024):.1f} MB")
        
        # 保存详细结果
        results_file = os.path.join(os.path.dirname(__file__), 'performance_test_results.json')
        results['test_config'] = {
            'test_range': test_range,
            'max_test_time': max_test_time,
            'test_time': datetime.now().isoformat()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {results_file}")
        
    finally:
        # 清理测试文件
        for test_file in [original_test_script, optimized_test_script]:
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"已清理测试文件: {test_file}")

def quick_benchmark():
    """快速基准测试"""
    print("快速基准测试 - 仅测试脚本启动和初始化时间")
    
    scripts = {
        '原始版本': r'd:\09_Code\Gis_Script\05_SAM\S1_GLDownload.py',
        '优化版本': r'd:\09_Code\Gis_Script\05_SAM\S1_GLDownload_Optimized.py'
    }
    
    for name, script_path in scripts.items():
        if os.path.exists(script_path):
            print(f"\n测试 {name}...")
            start_time = time.time()
            
            try:
                # 只导入模块，不执行主函数
                process = subprocess.Popen(
                    [sys.executable, '-c', f'import sys; sys.path.append(r"{os.path.dirname(script_path)}"); print("导入成功")'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=30)
                end_time = time.time()
                
                print(f"  启动时间: {end_time - start_time:.2f} 秒")
                print(f"  状态: {'成功' if process.returncode == 0 else '失败'}")
                if stderr:
                    print(f"  警告/错误: {stderr[:200]}...")
                    
            except Exception as e:
                print(f"  测试失败: {e}")
        else:
            print(f"\n{name} 脚本不存在: {script_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据下载性能测试')
    parser.add_argument('--quick', action='store_true', help='快速基准测试')
    parser.add_argument('--full', action='store_true', help='完整性能测试')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_benchmark()
    elif args.full:
        performance_test()
    else:
        print("请选择测试模式:")
        print("  --quick: 快速基准测试")
        print("  --full: 完整性能测试")
        
        choice = input("\n输入选择 (quick/full): ").strip().lower()
        if choice == 'quick':
            quick_benchmark()
        elif choice == 'full':
            performance_test()
        else:
            print("无效选择，退出程序")