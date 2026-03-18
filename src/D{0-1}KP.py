"""
D{0-1}背包问题求解程序
功能：采用动态规划算法求解折扣{0-1}背包问题
作者：软件工程实验
日期：2026年3月
"""

import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 常量定义 ====================

class DatasetType(Enum):
    """数据集类型枚举"""
    IDKP = "idkp"   # Inverse strongly correlated
    SDKP = "sdkp"   # Strongly correlated
    UDKP = "udkp"   # Uncorrelated
    WDKP = "wdkp"   # Weakly correlated


@dataclass
class Item:
    """单个物品类"""
    id: int              # 物品ID
    weight: float        # 重量
    value: float         # 价值
    set_id: int          # 所属项集ID
    
    @property
    def ratio(self) -> float:
        """价值重量比"""
        return self.value / self.weight if self.weight > 0 else 0


@dataclass
class ItemSet:
    """项集类（每组包含3个物品）"""
    set_id: int                    # 项集ID
    items: List[Item] = None       # 物品列表（3个）
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
    
    def add_item(self, item: Item):
        self.items.append(item)
    
    @property
    def discount_ratio(self) -> float:
        """第三项的价值重量比（用于排序）"""
        if len(self.items) >= 3:
            return self.items[2].ratio
        return 0.0


@dataclass
class Dataset:
    """D{0-1}KP数据集类"""
    type: DatasetType               # 数据集类型
    index: int                      # 数据集编号（1-10）
    name: str                       # 数据集名称
    capacity: float = 0.0            # 背包容量
    profits: List[float] = None      # 价值列表
    weights: List[float] = None      # 重量列表
    item_sets: List[ItemSet] = None  # 项集列表
    
    def __post_init__(self):
        if self.profits is None:
            self.profits = []
        if self.weights is None:
            self.weights = []
        if self.item_sets is None:
            self.item_sets = []
    
    @property
    def item_count(self) -> int:
        return len(self.profits)
    
    @property
    def set_count(self) -> int:
        return len(self.item_sets)
    
    def build_item_sets(self):
        """根据profits和weights构建项集"""
        self.item_sets = []
        n = min(len(self.profits), len(self.weights)) // 3 * 3
        
        for i in range(0, n, 3):
            set_id = i // 3
            item_set = ItemSet(set_id)
            
            for j in range(3):
                item = Item(
                    id=set_id * 3 + j,
                    weight=self.weights[i + j],
                    value=self.profits[i + j],
                    set_id=set_id
                )
                item_set.add_item(item)
            
            self.item_sets.append(item_set)
    
    def sort_by_discount_ratio(self, reverse: bool = True):
        """按项集第三项的价值重量比排序"""
        self.item_sets.sort(key=lambda x: x.discount_ratio, reverse=reverse)
    
    def get_weights_values(self) -> Tuple[List[float], List[float]]:
        """获取所有物品的重量和价值列表（用于绘图）"""
        return self.weights, self.profits
    
    def get_summary(self) -> str:
        return f"{self.name}: 项集数={self.set_count}, 物品数={self.item_count}, 背包容量={self.capacity:.0f}"


# ==================== 文件读取模块 ====================

class FileReader:
    """D{0-1}KP数据文件读取类"""
    
    BASE_PATH = r"D:\SoftwareProject\实验二 软件工程个人项目\Four kinds of D{0-1}KP instances"
    
    @classmethod
    def read_all_datasets(cls) -> Dict[str, List[Dataset]]:
        """读取所有四个数据集文件，每个文件包含10个数据集"""
        all_datasets = {'idkp': [], 'sdkp': [], 'udkp': [], 'wdkp': []}
        
        for dataset_type in DatasetType:
            file_path = os.path.join(cls.BASE_PATH, f"{dataset_type.value}1-10.txt")
            print(f"\n📖 正在读取: {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 按数据集分割
                pattern = rf'{dataset_type.value.upper()}\d+:'
                splits = re.split(pattern, content)
                
                for i, block in enumerate(splits[1:], 1):
                    dataset = cls._parse_dataset(block, dataset_type, i)
                    if dataset:
                        all_datasets[dataset_type.value].append(dataset)
                        print(f"  ✅ {dataset.name}: 容量={dataset.capacity:.0f}, 物品数={len(dataset.profits)}")
                        
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        
        return all_datasets
    
    @staticmethod
    def _parse_dataset(block: str, dtype: DatasetType, idx: int) -> Optional[Dataset]:
        """解析单个数据集"""
        lines = block.strip().split('\n')
        
        dataset = Dataset(
            type=dtype,
            index=idx,
            name=f"{dtype.value.upper()}{idx}"
        )
        
        profits = []
        weights = []
        current = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower = line.lower()
            
            # 解析容量
            if 'cubage' in lower or 'capacity' in lower:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    # 取最大的数字作为容量
                    dataset.capacity = max([int(n) for n in numbers])
                continue
            
            # 识别数据段
            if 'profit' in lower:
                current = 'profit'
                continue
            if 'weight' in lower:
                current = 'weight'
                continue
            
            # 解析数字
            if current == 'profit':
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if part and part.isdigit():
                        profits.append(int(part))
            
            elif current == 'weight':
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if part and part.isdigit():
                        weights.append(int(part))
        
        # 确保数据长度一致
        n = min(len(profits), len(weights))
        dataset.profits = profits[:n]
        dataset.weights = weights[:n]
        dataset.build_item_sets()
        
        return dataset


# ==================== 动态规划求解模块 ====================

class DKPSoLver:
    """D{0-1}KP求解器（分组背包动态规划）"""
    
    def solve(self, dataset: Dataset) -> Dict[str, Any]:
        """
        使用动态规划求解D{0-1}KP
        算法：分组背包动态规划（一维数组优化）
        """
        start_time = time.time()
        
        n_sets = len(dataset.item_sets)
        capacity = int(dataset.capacity)
        
        if n_sets == 0 or capacity <= 0:
            return {
                'max_value': 0,
                'capacity_used': 0,
                'solve_time': time.time() - start_time,
                'selected_items': [],
                'capacity': capacity,
                'dataset_name': dataset.name
            }
        
        # 初始化DP数组
        dp = [0] * (capacity + 1)
        choice = [[-1] * (capacity + 1) for _ in range(n_sets + 1)]
        
        # 动态规划主过程
        for i in range(1, n_sets + 1):
            item_set = dataset.item_sets[i-1]
            
            for j in range(capacity, -1, -1):
                best_value = dp[j]
                best_item = -1
                
                # 尝试选择当前组的每个物品
                for idx, item in enumerate(item_set.items):
                    w = int(item.weight)
                    if j >= w:
                        candidate = dp[j - w] + item.value
                        if candidate > best_value:
                            best_value = candidate
                            best_item = idx
                
                dp[j] = best_value
                choice[i][j] = best_item
        
        # 回溯找出选择了哪些物品
        selected_items = []
        remaining = capacity
        for i in range(n_sets, 0, -1):
            item_choice = choice[i][remaining]
            if item_choice != -1:
                item_set = dataset.item_sets[i-1]
                item = item_set.items[item_choice]
                selected_items.append({
                    'set_id': item_set.set_id,
                    'item_index': item_choice,
                    'weight': item.weight,
                    'value': item.value
                })
                remaining -= int(item.weight)
        
        end_time = time.time()
        
        return {
            'max_value': dp[capacity],
            'capacity_used': capacity - remaining,
            'solve_time': end_time - start_time,
            'selected_items': selected_items,
            'capacity': capacity,
            'dataset_name': dataset.name
        }
    
    def print_result(self, result: Dict):
        """打印求解结果"""
        print("\n" + "=" * 70)
        print(f"【{result['dataset_name']} 求解结果】")
        print("=" * 70)
        print(f"最大价值: {result['max_value']:,.2f}")
        print(f"使用容量: {result['capacity_used']:,.2f} / {result['capacity']:,.2f}")
        print(f"利用率: {result['capacity_used']/result['capacity']*100:.2f}%")
        print(f"求解时间: {result['solve_time']*1000:.2f} 毫秒")
        print(f"选中物品数: {len(result['selected_items'])}")
        
        if result['selected_items']:
            print("\n【选中物品(前10个)】")
            for i, item in enumerate(result['selected_items'][:10]):
                print(f"  项集{item['set_id']:3d} - 物品{item['item_index']}: "
                      f"重量={item['weight']:5.0f}, 价值={item['value']:5.0f}")
            
            total_weight = sum(item['weight'] for item in result['selected_items'])
            total_value = sum(item['value'] for item in result['selected_items'])
            print(f"\n【总计】")
            print(f"  总重量: {total_weight:,.2f}")
            print(f"  总价值: {total_value:,.2f}")
        print("=" * 70)


# ==================== 绘图模块 ====================

class DataPlotter:
    """数据绘图类"""
    
    @staticmethod
    def plot_scatter(dataset: Dataset, save_path: str = None):
        """
        绘制散点图：重量为横轴，价值为纵轴
        """
        weights, values = dataset.get_weights_values()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(weights, values, alpha=0.6, s=20, c='blue')
        plt.xlabel('重量', fontsize=12)
        plt.ylabel('价值', fontsize=12)
        plt.title(f'D{{0-1}}KP数据散点图 - {dataset.name} (容量={dataset.capacity:.0f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 散点图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_selected(dataset: Dataset, result: Dict, save_path: str = None):
        """
        绘制选中物品标记图
        """
        weights, values = dataset.get_weights_values()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(weights, values, alpha=0.2, s=20, c='gray', label='未选中')
        
        if result['selected_items']:
            sel_weights = [item['weight'] for item in result['selected_items']]
            sel_values = [item['value'] for item in result['selected_items']]
            plt.scatter(sel_weights, sel_values, alpha=0.8, s=50, c='red', label='选中')
        
        plt.xlabel('重量', fontsize=12)
        plt.ylabel('价值', fontsize=12)
        plt.title(f'选中物品标记 - {dataset.name} (价值={result["max_value"]:,.0f})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 标记图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_comparison(results: List[Dict], labels: List[str], save_path: str = None):
        """
        绘制多个结果的对比图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 价值对比
        values = [r['max_value'] for r in results]
        bars1 = ax1.bar(range(len(values)), values, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('数据集', fontsize=12)
        ax1.set_ylabel('最大价值', fontsize=12)
        ax1.set_title('各数据集最优价值对比', fontsize=14)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars1, values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 时间对比
        times = [r['solve_time'] * 1000 for r in results]
        bars2 = ax2.bar(range(len(times)), times, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('数据集', fontsize=12)
        ax2.set_ylabel('求解时间 (ms)', fontsize=12)
        ax2.set_title('各数据集求解时间对比', fontsize=14)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, t) in enumerate(zip(bars2, times)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('D{0-1}KP求解结果对比', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 对比图已保存: {save_path}")
        
        plt.show()


# ==================== 结果导出模块 ====================

class ResultExporter:
    """结果导出类"""
    
    @staticmethod
    def export_to_txt(result: Dict, file_path: str):
        """
        导出结果到txt文件
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"           D{{0-1}}KP求解结果报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"【数据集信息】\n")
            f.write(f"  数据集：{result['dataset_name']}\n")
            f.write(f"  背包容量：{result['capacity']:,.2f}\n\n")
            
            f.write(f"【求解结果】\n")
            f.write(f"  最大价值：{result['max_value']:,.2f}\n")
            f.write(f"  使用容量：{result['capacity_used']:,.2f}\n")
            f.write(f"  利用率：{result['capacity_used']/result['capacity']*100:.2f}%\n")
            f.write(f"  求解时间：{result['solve_time']*1000:.2f} 毫秒\n")
            f.write(f"  选中物品数：{len(result['selected_items'])}\n\n")
            
            if result['selected_items']:
                f.write(f"【选中物品详情】\n")
                f.write(f"{'项集ID':<8} {'物品索引':<8} {'重量':<10} {'价值':<10}\n")
                f.write("-" * 40 + "\n")
                
                total_weight = 0
                total_value = 0
                for item in sorted(result['selected_items'], key=lambda x: x['set_id']):
                    f.write(f"{item['set_id']:<8} {item['item_index']:<8} "
                           f"{item['weight']:<10.0f} {item['value']:<10.0f}\n")
                    total_weight += item['weight']
                    total_value += item['value']
                
                f.write("-" * 40 + "\n")
                f.write(f"{'总计':<18} {total_weight:<10.0f} {total_value:<10.0f}\n")
        
        print(f"💾 结果已保存到：{file_path}")
    
    @staticmethod
    def export_to_excel(results: List[Dict], dataset_names: List[str], file_path: str):
        """
        导出结果到Excel文件
        """
        # 汇总数据
        summary_data = []
        for i, (result, name) in enumerate(zip(results, dataset_names)):
            summary_data.append({
                '序号': i + 1,
                '数据集': name,
                '背包容量': result['capacity'],
                '最大价值': result['max_value'],
                '使用容量': result['capacity_used'],
                '利用率(%)': result['capacity_used']/result['capacity']*100,
                '求解时间(ms)': result['solve_time']*1000,
                '选中物品数': len(result['selected_items'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # 详细数据
        detail_data = []
        for i, (result, name) in enumerate(zip(results, dataset_names)):
            for item in result['selected_items']:
                detail_data.append({
                    '数据集序号': i + 1,
                    '数据集': name,
                    '项集ID': item['set_id'],
                    '物品索引': item['item_index'],
                    '重量': item['weight'],
                    '价值': item['value']
                })
        
        df_detail = pd.DataFrame(detail_data) if detail_data else pd.DataFrame()
        
        # 保存到Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='汇总结果', index=False)
            if not df_detail.empty:
                df_detail.to_excel(writer, sheet_name='选中物品详情', index=False)
        
        print(f"💾 结果已导出到Excel：{file_path}")


# ==================== 用户界面模块 ====================

class MenuSystem:
    """菜单系统 - 提供友好的用户界面"""
    
    def __init__(self):
        self.all_datasets = {}
        self.current_dataset = None
        self.results = {}
        self.base_path = r"D:\SoftwareProject\实验二 软件工程个人项目\Four kinds of D{0-1}KP instances"
        self.solver = DKPSoLver()
        self.plotter = DataPlotter()
        self.exporter = ResultExporter()
    
    def run(self):
        """运行主菜单"""
        self._print_header()
        self._load_all_data()
        
        while True:
            self._print_main_menu()
            choice = input("\n请输入您的选择 (1-9): ").strip()
            
            if choice == '1':
                self._select_dataset()
            elif choice == '2':
                self._plot_data()
            elif choice == '3':
                self._sort_data()
            elif choice == '4':
                self._solve_data()
            elif choice == '5':
                self._plot_selected()
            elif choice == '6':
                self._export_result()
            elif choice == '7':
                self._batch_process()
            elif choice == '8':
                self._list_datasets()
            elif choice == '9':
                print("\n感谢使用，再见！")
                break
            else:
                print("\n❌ 输入无效，请重新选择！")
            
            if choice != '9':
                input("\n按回车键继续...")
    
    def _print_header(self):
        print("\n" + "=" * 70)
        print("           D{0-1}背包问题求解程序 v3.0")
        print("=" * 70)
        print(f"数据目录: {self.base_path}")
        print("=" * 70)
    
    def _print_main_menu(self):
        print("\n【主菜单】")
        print("  1. 选择数据集")
        print("  2. 绘制散点图")
        print("  3. 按价值重量比排序")
        print("  4. 求解最优结果")
        print("  5. 绘制选中物品标记图")
        print("  6. 导出结果")
        print("  7. 批量处理")
        print("  8. 显示所有数据集")
        print("  9. 退出程序")
        
        if self.current_dataset:
            print(f"\n📌 当前数据集：{self.current_dataset.name}")
            print(f"   {self.current_dataset.get_summary()}")
    
    def _load_all_data(self):
        """加载所有数据集"""
        print("\n📁 正在加载数据集...")
        self.all_datasets = FileReader.read_all_datasets()
        
        total = sum(len(v) for v in self.all_datasets.values())
        print(f"\n✅ 成功加载 {total} 个数据集！")
    
    def _list_datasets(self):
        """显示所有数据集"""
        print("\n【所有数据集】")
        for type_name in ['idkp', 'sdkp', 'udkp', 'wdkp']:
            datasets = self.all_datasets.get(type_name, [])
            if datasets:
                print(f"\n{type_name.upper()}:")
                for d in datasets:
                    print(f"  {d.name}: 项集数={d.set_count:3d}, 容量={d.capacity:6.0f}")
    
    def _select_dataset(self):
        """选择数据集"""
        print("\n【选择数据集】")
        print("请选择类型：")
        print("  1. IDKP (Inverse strongly correlated)")
        print("  2. SDKP (Strongly correlated)")
        print("  3. UDKP (Uncorrelated)")
        print("  4. WDKP (Weakly correlated)")
        
        type_choice = input("请选择类型 (1-4): ").strip()
        type_map = {'1': 'idkp', '2': 'sdkp', '3': 'udkp', '4': 'wdkp'}
        
        if type_choice not in type_map:
            print("❌ 类型选择无效！")
            return
        
        datasets = self.all_datasets.get(type_map[type_choice], [])
        if not datasets:
            print(f"❌ 没有找到数据集！")
            return
        
        print(f"\n【{type_map[type_choice].upper()} 类型的数据集】")
        for i, d in enumerate(datasets, 1):
            print(f"  {i}. {d.name}: 项集数={d.set_count:3d}, 容量={d.capacity:6.0f}")
        
        try:
            idx = int(input(f"请选择数据集编号 (1-{len(datasets)}): ")) - 1
            if 0 <= idx < len(datasets):
                self.current_dataset = datasets[idx]
                print(f"✅ 已选择数据集：{self.current_dataset.name}")
            else:
                print("❌ 编号无效！")
        except:
            print("❌ 输入无效！")
    
    def _plot_data(self):
        """绘制散点图"""
        if not self.current_dataset:
            print("❌ 请先选择数据集！")
            return
        
        save = input("是否保存图片？(y/n, 默认n): ").strip().lower() == 'y'
        save_path = None
        if save:
            save_path = os.path.join(self.base_path, f"{self.current_dataset.name}_scatter.png")
        
        self.plotter.plot_scatter(self.current_dataset, save_path)
    
    def _plot_selected(self):
        """绘制选中物品标记图"""
        if not self.current_dataset:
            print("❌ 请先选择数据集！")
            return
        
        if self.current_dataset.name not in self.results:
            print("❌ 请先求解当前数据集！")
            return
        
        save = input("是否保存图片？(y/n, 默认n): ").strip().lower() == 'y'
        save_path = None
        if save:
            save_path = os.path.join(self.base_path, f"{self.current_dataset.name}_selected.png")
        
        self.plotter.plot_selected(self.current_dataset, self.results[self.current_dataset.name], save_path)
    
    def _sort_data(self):
        """按价值重量比排序"""
        if not self.current_dataset:
            print("❌ 请先选择数据集！")
            return
        
        print("\n排序前项集顺序（前5个）：")
        for i, s in enumerate(self.current_dataset.item_sets[:5]):
            print(f"  项集{i:3d}: 价值/重量比 = {s.discount_ratio:.4f}")
        
        order = input("排序方式（1-降序，2-升序，默认1）: ").strip()
        reverse = (order != '2')
        
        self.current_dataset.sort_by_discount_ratio(reverse)
        
        print("✅ 排序完成！")
        print("排序后项集顺序（前5个）：")
        arrow = "↓" if reverse else "↑"
        for i, s in enumerate(self.current_dataset.item_sets[:5]):
            print(f"  项集{i:3d}: 价值/重量比 = {s.discount_ratio:.4f} {arrow}")
    
    def _solve_data(self):
        """求解最优结果"""
        if not self.current_dataset:
            print("❌ 请先选择数据集！")
            return
        
        print("\n🔍 正在使用动态规划算法求解...")
        result = self.solver.solve(self.current_dataset)
        self.results[self.current_dataset.name] = result
        self.solver.print_result(result)
    
    def _export_result(self):
        """导出结果"""
        if not self.results:
            print("❌ 没有可导出的结果！")
            return
        
        print("\n【导出结果】")
        print("1. 导出当前数据集结果 (TXT)")
        print("2. 导出当前数据集结果 (Excel)")
        print("3. 导出所有结果 (Excel)")
        
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            if not self.current_dataset or self.current_dataset.name not in self.results:
                print("❌ 请先求解当前数据集！")
                return
            filename = f"{self.current_dataset.name}_result.txt"
            file_path = os.path.join(self.base_path, filename)
            self.exporter.export_to_txt(self.results[self.current_dataset.name], file_path)
        
        elif choice == '2':
            if not self.current_dataset or self.current_dataset.name not in self.results:
                print("❌ 请先求解当前数据集！")
                return
            filename = f"{self.current_dataset.name}_result.xlsx"
            file_path = os.path.join(self.base_path, filename)
            self.exporter.export_to_excel(
                [self.results[self.current_dataset.name]],
                [self.current_dataset.name],
                file_path
            )
        
        elif choice == '3':
            filename = "all_results.xlsx"
            file_path = os.path.join(self.base_path, filename)
            self.exporter.export_to_excel(
                list(self.results.values()),
                list(self.results.keys()),
                file_path
            )
        
        else:
            print("输入无效！")
    
    def _batch_process(self):
        """批量处理"""
        print("\n【批量处理】")
        print("请选择要处理的数据集类型：")
        print("  1. IDKP (Inverse strongly correlated)")
        print("  2. SDKP (Strongly correlated)")
        print("  3. UDKP (Uncorrelated)")
        print("  4. WDKP (Weakly correlated)")
        print("  5. 所有类型")
        
        type_choice = input("请选择 (1-5): ").strip()
        type_map = {
            '1': ['idkp'], '2': ['sdkp'], '3': ['udkp'], '4': ['wdkp'], '5': ['idkp', 'sdkp', 'udkp', 'wdkp']
        }
        
        if type_choice not in type_map:
            print("❌ 选择无效！")
            return
        
        datasets = []
        for typ in type_map[type_choice]:
            datasets.extend(self.all_datasets.get(typ, []))
        
        if not datasets:
            print("❌ 没有可处理的数据集！")
            return
        
        print(f"\n📊 正在处理 {len(datasets)} 个数据集...")
        
        results = []
        names = []
        
        for dataset in datasets:
            print(f"\n  {dataset.name}...")
            result = self.solver.solve(dataset)
            results.append(result)
            names.append(dataset.name)
            self.results[dataset.name] = result
            print(f"    最大价值: {result['max_value']:,.2f}, 时间: {result['solve_time']*1000:.2f}ms")
        
        # 显示汇总
        print("\n" + "=" * 70)
        print("【批量处理结果汇总】")
        print(f"{'序号':<4} {'数据集':<8} {'最大价值':<12} {'利用率':<8} {'时间(ms)':<10}")
        print("-" * 70)
        for i, (name, result) in enumerate(zip(names, results)):
            print(f"{i+1:<4} {name:<8} {result['max_value']:<12,.0f} "
                  f"{result['capacity_used']/result['capacity']*100:<8.2f}% "
                  f"{result['solve_time']*1000:<10.2f}")
        
        # 绘制对比图
        plot = input("\n是否绘制对比图？(y/n, 默认n): ").strip().lower() == 'y'
        if plot:
            save_path = os.path.join(self.base_path, "batch_comparison.png")
            self.plotter.plot_comparison(results, names, save_path)
        
        # 导出汇总
        export = input("是否导出汇总结果？(y/n, 默认n): ").strip().lower() == 'y'
        if export:
            file_path = os.path.join(self.base_path, "batch_result.xlsx")
            self.exporter.export_to_excel(results, names, file_path)


# ==================== 主程序入口 ====================

def main():
    """主函数"""
    # 检查必要库
    try:
        import matplotlib
        import pandas
    except ImportError as e:
        print("❌ 缺少必要库，请安装：")
        print("   pip install matplotlib pandas openpyxl")
        return
    
    # 运行菜单系统
    menu = MenuSystem()
    menu.run()


if __name__ == "__main__":
    main()