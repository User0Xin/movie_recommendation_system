"""
网络拓扑特性分析模块
分析用户-电影二部图的复杂网络特性
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
# 设置非交互式后端，适用于服务器环境（无图形界面）
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# 尝试导入GPU加速库（使用PyTorch，更容易安装）
GPU_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"✓ 检测到PyTorch GPU支持 (CUDA {torch.version.cuda})")
    else:
        print("⚠ PyTorch已安装，但未检测到GPU设备")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    print("⚠ PyTorch未安装，将使用CPU计算")
    print("  如需GPU加速，请安装: pip install torch")

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import FILE_PATH

# 使用最基础的字体设置，确保文字能正常显示
# 直接使用英文标签，避免字体问题
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
CHINESE_FONT_AVAILABLE = False  # 强制使用英文标签

warnings.filterwarnings('ignore')


class NetworkAnalyzer:
    """网络拓扑特性分析器"""
    
    def __init__(self, ratings_path=None, movies_path=None):
        """
        初始化网络分析器
        
        Args:
            ratings_path: 评分数据路径
            movies_path: 电影数据路径
        """
        if ratings_path is None:
            ratings_path = FILE_PATH / "ratings.dat"
        if movies_path is None:
            movies_path = FILE_PATH / "movies.dat"
            
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.G = None
        self.user_nodes = None
        self.movie_nodes = None
        self.stats = {}
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        # 加载评分数据
        self.ratings = pd.read_csv(
            self.ratings_path,
            sep='::',
            engine='python',
            names=['userId', 'movieId', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        # 加载电影数据
        self.movies = pd.read_csv(
            self.movies_path,
            sep='::',
            engine='python',
            names=['movieId', 'title', 'genres'],
            encoding='latin-1'
        )
        print(f"加载完成: {len(self.ratings)} 条评分记录, {len(self.movies)} 部电影")
        
    def build_bipartite_graph(self):
        """构建用户-电影二部图"""
        print("正在构建二部图...")
        self.G = nx.Graph()
        
        # 添加节点和边
        for _, row in self.ratings.iterrows():
            user_node = f"U_{row['userId']}"
            movie_node = f"M_{row['movieId']}"
            
            # 添加节点
            self.G.add_node(user_node, node_type='user', id=row['userId'])
            self.G.add_node(movie_node, node_type='movie', id=row['movieId'])
            
            # 添加边（权重为评分）
            self.G.add_edge(user_node, movie_node, weight=row['rating'])
        
        # 分离用户节点和电影节点
        self.user_nodes = [n for n in self.G.nodes() if n.startswith('U_')]
        self.movie_nodes = [n for n in self.G.nodes() if n.startswith('M_')]
        
        print(f"图构建完成: {self.G.number_of_nodes()} 个节点, {self.G.number_of_edges()} 条边")
        print(f"用户节点: {len(self.user_nodes)}, 电影节点: {len(self.movie_nodes)}")
        
    def calculate_basic_stats(self):
        """计算基本统计特性"""
        print("\n=== 计算基本统计特性 ===")
        
        # 节点数和边数
        self.stats['num_nodes'] = self.G.number_of_nodes()
        self.stats['num_edges'] = self.G.number_of_edges()
        self.stats['num_users'] = len(self.user_nodes)
        self.stats['num_movies'] = len(self.movie_nodes)
        
        # 平均度
        degrees = dict(self.G.degree())
        self.stats['avg_degree'] = np.mean(list(degrees.values()))
        self.stats['max_degree'] = max(degrees.values())
        self.stats['min_degree'] = min(degrees.values())
        
        # 度分布
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        self.stats['degree_sequence'] = degree_sequence
        
        print(f"节点数: {self.stats['num_nodes']}")
        print(f"边数: {self.stats['num_edges']}")
        print(f"平均度: {self.stats['avg_degree']:.2f}")
        print(f"最大度: {self.stats['max_degree']}")
        print(f"最小度: {self.stats['min_degree']}")
        
    def _compute_clustering_worker(self, args):
        """多进程工作函数：计算单个投影图的聚类系数"""
        node_list, edges_data, node_type = args
        try:
            # 重建子图
            G_sub = nx.Graph()
            G_sub.add_nodes_from(node_list)
            for edge in edges_data:
                G_sub.add_edge(edge[0], edge[1])
            
            # 投影并计算聚类系数
            if node_type == 'user':
                projection = nx.bipartite.projected_graph(G_sub, node_list)
            else:
                projection = nx.bipartite.projected_graph(G_sub, node_list)
            
            clustering = nx.average_clustering(projection)
            return (node_type, clustering, None)
        except Exception as e:
            return (node_type, None, str(e))
    
    def _compute_clustering_gpu_torch(self, nx_graph, node_list, node_type):
        """使用PyTorch GPU加速计算聚类系数"""
        if not GPU_AVAILABLE or not TORCH_AVAILABLE:
            return None
        
        try:
            print(f"  🚀 使用PyTorch GPU加速计算{node_type}网络聚类系数...")
            sys.stdout.flush()
            
            # 将图转换为邻接矩阵（使用PyTorch）
            nodes = list(nx_graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            n = len(nodes)
            
            # 创建邻接矩阵（在GPU上）
            device = torch.device('cuda')
            adj_matrix = torch.zeros((n, n), dtype=torch.float32, device=device)
            
            # 填充邻接矩阵
            for u, v in nx_graph.edges():
                i, j = node_to_idx[u], node_to_idx[v]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # 无向图
            
            # 计算度矩阵
            degrees = torch.sum(adj_matrix, dim=1)
            
            # 计算 A^3 的对角线元素（三角形数量）
            # 对于节点i，A^3[i,i] = 三角形数量 * 2（每个三角形被计算两次）
            adj_cubed = torch.mm(torch.mm(adj_matrix, adj_matrix), adj_matrix)
            triangles = torch.diag(adj_cubed) / 2.0  # 除以2因为每个三角形被计算两次
            
            # 计算每个节点的聚类系数
            # C_i = 2 * triangles_i / (k_i * (k_i - 1))
            # 避免除零
            k = degrees.float()
            k_safe = torch.clamp(k * (k - 1), min=1.0)
            clustering_per_node = 2.0 * triangles / k_safe
            
            # 只对度 >= 2 的节点计算平均值
            valid_mask = k >= 2
            if torch.sum(valid_mask) > 0:
                avg_clustering = torch.mean(clustering_per_node[valid_mask]).item()
                print(f"  ✓ PyTorch GPU计算完成")
                sys.stdout.flush()
                return float(avg_clustering)
            else:
                return None
                
        except Exception as e:
            print(f"  PyTorch GPU计算失败: {e}，回退到CPU")
            sys.stdout.flush()
            return None
    
    def calculate_clustering_coefficient(self, use_sampling=True, sample_size=2000, num_processes=None, use_gpu=None):
        """计算聚类系数（支持GPU加速和多进程并行计算）
        
        Args:
            use_sampling: 是否使用采样方法（对于大型网络）
            sample_size: 采样节点数量
            num_processes: 进程数（None表示使用CPU核心数）
            use_gpu: 是否使用GPU（None表示自动检测）
        """
        print("\n=== 计算聚类系数 ===")
        sys.stdout.flush()  # 强制刷新输出
        
        # 检测GPU可用性
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        
        if use_gpu and GPU_AVAILABLE:
            print("🚀 使用GPU加速计算")
        else:
            if use_gpu and not GPU_AVAILABLE:
                print("⚠ GPU不可用，回退到CPU计算")
            else:
                print("💻 使用CPU计算")
        
        if num_processes is None:
            num_processes = max(1, cpu_count() - 1)  # 保留一个核心给系统
        
        num_nodes = self.G.number_of_nodes()
        print(f"网络节点数: {num_nodes}")
        if not use_gpu:
            print(f"使用进程数: {num_processes}")
        sys.stdout.flush()
        
        # 对于大型网络，使用采样方法以提高速度
        if use_sampling and num_nodes > 5000:
            print(f"网络较大，使用采样方法（采样 {sample_size} 个节点）...")
            print("这可能需要几分钟时间，请耐心等待...")
            sys.stdout.flush()
            
            try:
                # 准备用户网络数据
                print("\n[1/2] 正在准备用户网络数据并计算聚类系数...")
                sys.stdout.flush()
                start_time = time.time()
                
                if len(self.user_nodes) > sample_size:
                    sampled_users = list(np.random.choice(self.user_nodes, sample_size, replace=False))
                    print(f"  采样了 {len(sampled_users)} 个用户节点")
                else:
                    sampled_users = self.user_nodes
                    print(f"  使用全部 {len(sampled_users)} 个用户节点")
                
                sys.stdout.flush()
                
                # 构建子图（只包含采样节点和所有电影节点）
                print("  正在构建用户投影网络...")
                sys.stdout.flush()
                user_subgraph = self.G.subgraph(list(sampled_users) + self.movie_nodes)
                
                print("  正在计算用户投影网络聚类系数（这可能需要一些时间）...")
                sys.stdout.flush()
                user_projection = nx.bipartite.projected_graph(user_subgraph, sampled_users)
                
                # 尝试使用GPU计算（PyTorch）
                if use_gpu and GPU_AVAILABLE and TORCH_AVAILABLE:
                    gpu_result = self._compute_clustering_gpu_torch(user_projection, sampled_users, 'User')
                    if gpu_result is not None:
                        global_clustering = gpu_result
                    else:
                        print("  使用CPU计算平均聚类系数...")
                        sys.stdout.flush()
                        global_clustering = nx.average_clustering(user_projection)
                else:
                    print("  正在计算平均聚类系数...")
                    sys.stdout.flush()
                    global_clustering = nx.average_clustering(user_projection)
                
                elapsed = time.time() - start_time
                
                self.stats['global_clustering_user'] = global_clustering
                print(f"  ✓ 用户投影网络全局聚类系数: {global_clustering:.4f} (耗时: {elapsed:.1f}秒)")
                sys.stdout.flush()
                
                # 准备电影网络数据
                print("\n[2/2] 正在准备电影网络数据并计算聚类系数...")
                sys.stdout.flush()
                start_time = time.time()
                
                if len(self.movie_nodes) > sample_size:
                    sampled_movies = list(np.random.choice(self.movie_nodes, sample_size, replace=False))
                    print(f"  采样了 {len(sampled_movies)} 个电影节点")
                else:
                    sampled_movies = self.movie_nodes
                    print(f"  使用全部 {len(sampled_movies)} 个电影节点")
                
                sys.stdout.flush()
                
                # 构建子图
                print("  正在构建电影投影网络...")
                sys.stdout.flush()
                movie_subgraph = self.G.subgraph(list(sampled_movies) + self.user_nodes)
                
                print("  正在计算电影投影网络聚类系数（这可能需要一些时间）...")
                sys.stdout.flush()
                movie_projection = nx.bipartite.projected_graph(movie_subgraph, sampled_movies)
                
                # 尝试使用GPU计算（PyTorch）
                if use_gpu and GPU_AVAILABLE and TORCH_AVAILABLE:
                    gpu_result = self._compute_clustering_gpu_torch(movie_projection, sampled_movies, 'Movie')
                    if gpu_result is not None:
                        global_clustering_movie = gpu_result
                    else:
                        print("  使用CPU计算平均聚类系数...")
                        sys.stdout.flush()
                        global_clustering_movie = nx.average_clustering(movie_projection)
                else:
                    print("  正在计算平均聚类系数...")
                    sys.stdout.flush()
                    global_clustering_movie = nx.average_clustering(movie_projection)
                
                elapsed = time.time() - start_time
                
                self.stats['global_clustering_movie'] = global_clustering_movie
                print(f"  ✓ 电影投影网络全局聚类系数: {global_clustering_movie:.4f} (耗时: {elapsed:.1f}秒)")
                sys.stdout.flush()
                    
            except Exception as e:
                print(f"计算聚类系数时出错: {e}")
                print("  尝试使用更小的采样...")
                sys.stdout.flush()
                try:
                    # 使用更小的采样（如果原始采样失败，尝试更小的采样）
                    # 使用原始 sample_size 的一半，但不超过 2000，至少 100
                    small_sample = min(2000, max(100, sample_size // 2))
                    if len(self.user_nodes) > small_sample:
                        sampled_users = list(np.random.choice(self.user_nodes, small_sample, replace=False))
                        user_subgraph = self.G.subgraph(list(sampled_users) + self.movie_nodes)
                        user_projection = nx.bipartite.projected_graph(user_subgraph, sampled_users)
                        global_clustering = nx.average_clustering(user_projection)
                        self.stats['global_clustering_user'] = global_clustering
                        print(f"  ✓ 用户网络聚类系数: {global_clustering:.4f} (小采样)")
                    else:
                        self.stats['global_clustering_user'] = 0
                    
                    if len(self.movie_nodes) > small_sample:
                        sampled_movies = list(np.random.choice(self.movie_nodes, small_sample, replace=False))
                        movie_subgraph = self.G.subgraph(list(sampled_movies) + self.user_nodes)
                        movie_projection = nx.bipartite.projected_graph(movie_subgraph, sampled_movies)
                        global_clustering_movie = nx.average_clustering(movie_projection)
                        self.stats['global_clustering_movie'] = global_clustering_movie
                        print(f"  ✓ 电影网络聚类系数: {global_clustering_movie:.4f} (小采样)")
                    else:
                        self.stats['global_clustering_movie'] = 0
                except Exception as e2:
                    print(f"  使用小采样也失败: {e2}")
                    self.stats['global_clustering_user'] = 0
                    self.stats['global_clustering_movie'] = 0
                sys.stdout.flush()
        else:
            # 小网络或禁用采样时，直接使用全部节点计算
            if not use_sampling:
                print("  使用全部节点计算（禁用采样模式）...")
            else:
                print(f"  网络节点数 ({num_nodes}) <= 5000，使用全部节点计算...")
            sys.stdout.flush()
            
            try:
                print("  正在计算用户投影网络聚类系数...")
                sys.stdout.flush()
                start_time = time.time()
                user_projection = nx.bipartite.projected_graph(self.G, self.user_nodes)
                
                # 尝试使用GPU计算
                if use_gpu and GPU_AVAILABLE and TORCH_AVAILABLE:
                    gpu_result = self._compute_clustering_gpu_torch(user_projection, self.user_nodes, 'User')
                    if gpu_result is not None:
                        global_clustering = gpu_result
                    else:
                        global_clustering = nx.average_clustering(user_projection)
                else:
                    global_clustering = nx.average_clustering(user_projection)
                
                elapsed = time.time() - start_time
                self.stats['global_clustering_user'] = global_clustering
                print(f"  ✓ 用户投影网络全局聚类系数: {global_clustering:.4f} (耗时: {elapsed:.1f}秒)")
                sys.stdout.flush()
                
                print("  正在计算电影投影网络聚类系数...")
                sys.stdout.flush()
                start_time = time.time()
                movie_projection = nx.bipartite.projected_graph(self.G, self.movie_nodes)
                
                # 尝试使用GPU计算
                if use_gpu and GPU_AVAILABLE and TORCH_AVAILABLE:
                    gpu_result = self._compute_clustering_gpu_torch(movie_projection, self.movie_nodes, 'Movie')
                    if gpu_result is not None:
                        global_clustering_movie = gpu_result
                    else:
                        global_clustering_movie = nx.average_clustering(movie_projection)
                else:
                    global_clustering_movie = nx.average_clustering(movie_projection)
                
                elapsed = time.time() - start_time
                self.stats['global_clustering_movie'] = global_clustering_movie
                print(f"  ✓ 电影投影网络全局聚类系数: {global_clustering_movie:.4f} (耗时: {elapsed:.1f}秒)")
                sys.stdout.flush()
            except Exception as e:
                print(f"计算聚类系数时出错: {e}")
                import traceback
                traceback.print_exc()
                self.stats['global_clustering_user'] = None
                self.stats['global_clustering_movie'] = None
                sys.stdout.flush()
        
        print("聚类系数计算完成！\n")
        sys.stdout.flush()
    
    def calculate_path_length(self):
        """计算路径长度"""
        print("\n=== 计算路径长度 ===")
        
        # 对于大型网络，使用采样方法
        if self.G.number_of_nodes() > 5000:
            print("网络较大，使用采样方法计算路径长度...")
            sample_size = min(2000, len(self.user_nodes))
            sampled_users = np.random.choice(self.user_nodes, sample_size, replace=False)
            
            path_lengths = []
            for i, u1 in enumerate(sampled_users):
                if i % 100 == 0:
                    print(f"  处理进度: {i}/{sample_size}")
                for u2 in sampled_users[i+1:]:
                    try:
                        path = nx.shortest_path_length(self.G, u1, u2)
                        path_lengths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
            if path_lengths:
                self.stats['avg_path_length'] = np.mean(path_lengths)
                self.stats['diameter'] = max(path_lengths)
                print(f"平均路径长度: {self.stats['avg_path_length']:.2f}")
                print(f"网络直径: {self.stats['diameter']}")
            else:
                print("无法计算路径长度（网络可能不连通）")
                self.stats['avg_path_length'] = None
                self.stats['diameter'] = None
        else:
            # 小网络直接计算
            try:
                self.stats['avg_path_length'] = nx.average_shortest_path_length(self.G)
                self.stats['diameter'] = nx.diameter(self.G)
                print(f"平均路径长度: {self.stats['avg_path_length']:.2f}")
                print(f"网络直径: {self.stats['diameter']}")
            except nx.NetworkXError as e:
                print(f"网络不连通: {e}")
                # 计算最大连通子图的路径长度
                largest_cc = max(nx.connected_components(self.G), key=len)
                subgraph = self.G.subgraph(largest_cc)
                self.stats['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                self.stats['diameter'] = nx.diameter(subgraph)
                print(f"最大连通子图平均路径长度: {self.stats['avg_path_length']:.2f}")
                print(f"最大连通子图直径: {self.stats['diameter']}")
    
    def analyze_degree_distribution(self):
        """分析度分布（检测幂律分布）"""
        print("\n=== 分析度分布 ===")
        
        degrees = dict(self.G.degree())
        degree_values = list(degrees.values())
        
        # 度分布统计
        unique_degrees, counts = np.unique(degree_values, return_counts=True)
        degree_dist = dict(zip(unique_degrees, counts))
        
        self.stats['degree_distribution'] = degree_dist
        
        # 幂律分布检测
        # 使用与可视化相同的直接度分布方法，确保结果一致
        # 只考虑度 >= 1 的节点
        unique_degrees_filtered = unique_degrees[unique_degrees >= 1]
        counts_filtered = counts[unique_degrees >= 1]
        
        if len(unique_degrees_filtered) > 10:
            try:
                # 在对数空间进行线性拟合
                # 对于幂律分布：count = C * degree^(-gamma)
                # 所以 log(count) = log(C) - gamma * log(degree)
                log_degrees = np.log(unique_degrees_filtered)
                log_counts = np.log(counts_filtered)
                
                # 去除无穷大和NaN
                valid_mask = np.isfinite(log_degrees) & np.isfinite(log_counts) & (log_counts > -np.inf)
                if np.sum(valid_mask) > 5:
                    log_degrees_valid = log_degrees[valid_mask]
                    log_counts_valid = log_counts[valid_mask]
                    
                    # 线性拟合：log(count) = a * log(degree) + b
                    # 斜率应该是负数，gamma = -slope
                    slope, intercept = np.polyfit(log_degrees_valid, log_counts_valid, 1)
                    power_law_exponent = -slope  # 幂律指数（应该是正数）
                    
                    self.stats['power_law_exponent'] = power_law_exponent
                    self.stats['is_power_law'] = power_law_exponent > 1.0  # 幂律指数通常 > 1.0
                    
                    print(f"幂律指数 (gamma): {power_law_exponent:.4f}")
                    print(f"是否为幂律分布: {self.stats['is_power_law']}")
                else:
                    self.stats['power_law_exponent'] = None
                    self.stats['is_power_law'] = False
            except Exception as e:
                print(f"幂律拟合失败: {e}")
                self.stats['power_law_exponent'] = None
                self.stats['is_power_law'] = False
        else:
            self.stats['power_law_exponent'] = None
            self.stats['is_power_law'] = False
    
    def calculate_degree_correlation(self):
        """计算度相关性（同配性/异配性）"""
        print("\n=== 计算度相关性 ===")
        
        try:
            # 计算度相关性（Pearson相关系数）
            degrees = dict(self.G.degree())
            edges = list(self.G.edges())
            
            if len(edges) > 0:
                edge_degrees = [(degrees[u], degrees[v]) for u, v in edges]
                degrees_u = [d[0] for d in edge_degrees]
                degrees_v = [d[1] for d in edge_degrees]
                
                correlation = np.corrcoef(degrees_u, degrees_v)[0, 1]
                self.stats['degree_correlation'] = correlation
                
                if correlation > 0:
                    assortativity_type = "同配性 (Assortative)"
                elif correlation < 0:
                    assortativity_type = "异配性 (Disassortative)"
                else:
                    assortativity_type = "中性 (Neutral)"
                
                print(f"度相关性: {correlation:.4f} ({assortativity_type})")
            else:
                self.stats['degree_correlation'] = None
        except Exception as e:
            print(f"计算度相关性时出错: {e}")
            self.stats['degree_correlation'] = None
    
    def identify_key_nodes(self, top_k=10):
        """识别关键节点（高影响力用户/电影）"""
        print(f"\n=== 识别Top-{top_k}关键节点 ===")
        
        degrees = dict(self.G.degree())
        
        # 按度排序
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        
        # 用户节点
        user_key_nodes = [(n, d) for n, d in sorted_nodes if n.startswith('U_')][:top_k]
        # 电影节点
        movie_key_nodes = [(n, d) for n, d in sorted_nodes if n.startswith('M_')][:top_k]
        
        self.stats['top_users'] = user_key_nodes
        self.stats['top_movies'] = movie_key_nodes
        
        print(f"\nTop-{top_k} 高影响力用户:")
        for i, (node, degree) in enumerate(user_key_nodes, 1):
            user_id = node.replace('U_', '')
            print(f"  {i}. 用户 {user_id}: 度 = {degree}")
        
        print(f"\nTop-{top_k} 高影响力电影:")
        for i, (node, degree) in enumerate(movie_key_nodes, 1):
            movie_id = node.replace('M_', '')
            # 查找电影名称
            movie_info = self.movies[self.movies['movieId'] == int(movie_id)]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                print(f"  {i}. {title} (ID: {movie_id}): 度 = {degree}")
            else:
                print(f"  {i}. 电影 {movie_id}: 度 = {degree}")
    
    def visualize_degree_distribution(self, save_path='network_degree_distribution.png'):
        """可视化度分布"""
        print(f"\n=== 可视化度分布 ===")
        
        # 直接使用英文标签，确保在任何环境下都能正常显示
        labels = {
            'degree': 'Degree',
            'count': 'Node Count',
            'hist_title': 'Degree Distribution Histogram',
            'loglog_title': 'Degree Distribution (Log-Log Plot)',
            'loglog_x': 'Degree (Log Scale)',
            'loglog_y': 'Node Count (Log Scale)',
            'compare_title': 'User vs Movie Degree Distribution',
            'user': 'User',
            'movie': 'Movie',
            'ccdf_title': 'Complementary Cumulative Distribution (CCDF)',
            'ccdf_y': 'P(X >= k) (Log Scale)',
            'fit_label': 'Power Law Fit'
        }
        
        degrees = dict(self.G.degree())
        degree_values = list(degrees.values())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 度分布直方图
        ax1 = axes[0, 0]
        ax1.hist(degree_values, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(labels['degree'], fontsize=12)
        ax1.set_ylabel(labels['count'], fontsize=12)
        ax1.set_title(labels['hist_title'], fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 对数-对数度分布（检测幂律）
        ax2 = axes[0, 1]
        unique_degrees, counts = np.unique(degree_values, return_counts=True)
        # 过滤掉0度节点
        valid_idx = unique_degrees > 0
        unique_degrees = unique_degrees[valid_idx]
        counts = counts[valid_idx]
        
        ax2.loglog(unique_degrees, counts, 'bo', markersize=4, alpha=0.6, label='Data')
        ax2.set_xlabel(labels['loglog_x'], fontsize=12)
        ax2.set_ylabel(labels['loglog_y'], fontsize=12)
        ax2.set_title(labels['loglog_title'], fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        # 如果有幂律指数，绘制拟合线
        # 使用存储的gamma值，确保与报告中一致
        if self.stats.get('power_law_exponent') is not None:
            gamma = self.stats['power_law_exponent']
            print(f"  绘制幂律拟合线，指数 γ = {gamma:.4f} (使用存储的值)")
            
            # 使用存储的gamma值计算拟合线
            # 对于幂律分布：count = C * degree^(-gamma)
            # 所以 log(count) = log(C) - gamma * log(degree)
            # 我们需要根据数据点计算截距 log(C)
            log_degrees = np.log(unique_degrees[unique_degrees > 0])
            log_counts = np.log(counts[unique_degrees > 0])
            
            # 去除无穷大和NaN
            valid_mask = np.isfinite(log_degrees) & np.isfinite(log_counts) & (log_counts > -np.inf)
            if np.sum(valid_mask) > 5:
                log_degrees_valid = log_degrees[valid_mask]
                log_counts_valid = log_counts[valid_mask]
                
                # 使用存储的gamma值，计算截距
                # log(count) = log(C) - gamma * log(degree)
                # 所以 log(C) = log(count) + gamma * log(degree)
                # 使用最小二乘法计算最佳截距
                log_C_values = log_counts_valid + gamma * log_degrees_valid
                log_C = np.mean(log_C_values)  # 使用平均值作为截距
                intercept = log_C
                
                # 生成拟合线
                # 使用数据点的实际范围
                x_min = unique_degrees[valid_mask].min()
                x_max = unique_degrees[valid_mask].max()
                x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 200)
                
                # 计算拟合线：y = exp(intercept) * x^(-gamma)
                y_fit = np.exp(intercept) * (x_fit ** (-gamma))
                
                # 只移除明显异常的值（负值或无穷大）
                valid_fit_mask = (y_fit > 0) & np.isfinite(y_fit) & (y_fit <= counts.max() * 50)
                
                if np.sum(valid_fit_mask) > 10:
                    # 绘制拟合线（只绘制有效部分）
                    ax2.plot(x_fit[valid_fit_mask], y_fit[valid_fit_mask], 'r--', linewidth=2.5, 
                            label=f"{labels['fit_label']} (γ={gamma:.2f})")
                    ax2.legend(fontsize=10)
                else:
                    # 如果过滤后点太少，直接绘制全部（不裁剪）
                    ax2.plot(x_fit, y_fit, 'r--', linewidth=2.5, 
                            label=f"{labels['fit_label']} (γ={gamma:.2f})")
                    ax2.legend(fontsize=10)
            else:
                # 如果数据不足，使用简化的拟合线
                print("  使用简化的拟合线")
                x_fit = np.logspace(np.log10(unique_degrees.min()), 
                                  np.log10(unique_degrees.max()), 200)
                # 使用统计的gamma值
                # count = C * degree^(-gamma)，所以拟合线应该向下倾斜
                # 从数据中估计常数C：在最小度值处，count应该接近counts.max()
                C = counts.max() * (unique_degrees.min() ** gamma)
                y_fit = C * (x_fit ** (-gamma))
                # 只移除异常值，不要过度裁剪
                valid_fit = (y_fit > 0) & np.isfinite(y_fit)
                ax2.plot(x_fit[valid_fit], y_fit[valid_fit], 'r--', linewidth=2.5, 
                        label=f"{labels['fit_label']} (γ={gamma:.2f})")
                ax2.legend(fontsize=10)
        else:
            print("  未计算幂律指数，跳过拟合线绘制")
        
        # 3. 用户和电影度分布对比
        ax3 = axes[1, 0]
        user_degrees = [degrees[n] for n in self.user_nodes]
        movie_degrees = [degrees[n] for n in self.movie_nodes]
        
        ax3.hist(user_degrees, bins=30, alpha=0.6, label=labels['user'], color='blue', edgecolor='black')
        ax3.hist(movie_degrees, bins=30, alpha=0.6, label=labels['movie'], color='red', edgecolor='black')
        ax3.set_xlabel(labels['degree'], fontsize=12)
        ax3.set_ylabel(labels['count'], fontsize=12)
        ax3.set_title(labels['compare_title'], fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 累积度分布（CCDF）
        ax4 = axes[1, 1]
        # 正确计算CCDF：对于每个唯一的度值k，计算度值>=k的节点比例
        sorted_degrees = np.sort(degree_values)
        unique_degrees_sorted = np.unique(sorted_degrees)
        
        # 对于每个唯一的度值，计算有多少节点的度值 >= 该度值
        ccdf_values = []
        for k in unique_degrees_sorted:
            count_ge_k = np.sum(sorted_degrees >= k)
            ccdf_k = count_ge_k / len(sorted_degrees)
            ccdf_values.append(ccdf_k)
        
        ccdf_values = np.array(ccdf_values)
        
        # 过滤掉CCDF为0的值（在对数空间中无法显示）
        valid_ccdf_mask = ccdf_values > 0
        unique_degrees_ccdf = unique_degrees_sorted[valid_ccdf_mask]
        ccdf_values = ccdf_values[valid_ccdf_mask]
        
        ax4.loglog(unique_degrees_ccdf, ccdf_values, 'b-', linewidth=2, alpha=0.7, label='CCDF')
        
        # 如果有幂律指数，绘制理论CCDF拟合线
        # 如果 P(k) ~ k^(-γ)，则 CCDF(k) ~ k^(-γ+1)
        if self.stats.get('power_law_exponent') is not None:
            gamma = self.stats['power_law_exponent']
            ccdf_exponent = -(gamma - 1)  # CCDF的幂律指数 = -(γ-1)
            
            print(f"  绘制CCDF理论拟合线，CCDF指数 = {ccdf_exponent:.4f} (基于 γ={gamma:.4f})")
            
            # 计算CCDF拟合线的截距
            if len(unique_degrees_ccdf) > 5:
                log_degrees_ccdf = np.log(unique_degrees_ccdf)
                log_ccdf = np.log(ccdf_values)
                
                valid_ccdf_fit_mask = np.isfinite(log_degrees_ccdf) & np.isfinite(log_ccdf) & (log_ccdf > -np.inf)
                if np.sum(valid_ccdf_fit_mask) > 5:
                    log_degrees_ccdf_valid = log_degrees_ccdf[valid_ccdf_fit_mask]
                    log_ccdf_valid = log_ccdf[valid_ccdf_fit_mask]
                    
                    # 使用理论指数计算截距：log(CCDF) = log(C) + ccdf_exponent * log(degree)
                    # 所以 log(C) = log(CCDF) - ccdf_exponent * log(degree)
                    # 使用加权平均，给中间区域（更符合幂律的区域）更高权重
                    # 中间区域通常是度值的中位数附近，这些区域更符合幂律假设
                    median_idx = len(log_degrees_ccdf_valid) // 2
                    # 使用高斯权重，中心区域权重更高
                    indices = np.arange(len(log_degrees_ccdf_valid))
                    weights = np.exp(-0.5 * ((indices - median_idx) / (len(log_degrees_ccdf_valid) / 4)) ** 2)
                    weights = weights / weights.sum()  # 归一化权重
                    
                    log_C_ccdf_values = log_ccdf_valid - ccdf_exponent * log_degrees_ccdf_valid
                    log_C_ccdf = np.average(log_C_ccdf_values, weights=weights)  # 使用加权平均作为截距
                    
                    # 生成拟合线
                    x_min_ccdf = unique_degrees_ccdf[valid_ccdf_fit_mask].min()
                    x_max_ccdf = unique_degrees_ccdf[valid_ccdf_fit_mask].max()
                    x_fit_ccdf = np.logspace(np.log10(x_min_ccdf), np.log10(x_max_ccdf), 200)
                    
                    # CCDF拟合线：y = exp(log_C_ccdf) * x^(ccdf_exponent)
                    y_fit_ccdf = np.exp(log_C_ccdf) * (x_fit_ccdf ** ccdf_exponent)
                    
                    # 过滤有效值
                    valid_fit_ccdf_mask = (y_fit_ccdf > 0) & np.isfinite(y_fit_ccdf) & (y_fit_ccdf <= 1.0)
                    
                    if np.sum(valid_fit_ccdf_mask) > 10:
                        ax4.plot(x_fit_ccdf[valid_fit_ccdf_mask], y_fit_ccdf[valid_fit_ccdf_mask], 
                                'r--', linewidth=2.5, alpha=0.8,
                                label=f'CCDF Fit (exp={ccdf_exponent:.2f})')
                        ax4.legend(fontsize=10)
        
        ax4.set_xlabel(labels['loglog_x'], fontsize=12)
        ax4.set_ylabel(labels['ccdf_y'], fontsize=12)
        ax4.set_title(labels['ccdf_title'], fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"度分布图已保存到: {save_path}")
        plt.close()
    
    def generate_report(self, save_path='network_analysis_report.txt'):
        """生成网络分析报告"""
        print(f"\n=== 生成分析报告 ===")
        
        report = []
        report.append("=" * 60)
        report.append("网络拓扑特性分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 基本统计
        report.append("【基本统计特性】")
        report.append(f"节点总数: {self.stats['num_nodes']}")
        report.append(f"边总数: {self.stats['num_edges']}")
        report.append(f"用户节点数: {self.stats['num_users']}")
        report.append(f"电影节点数: {self.stats['num_movies']}")
        report.append(f"平均度: {self.stats['avg_degree']:.4f}")
        report.append(f"最大度: {self.stats['max_degree']}")
        report.append(f"最小度: {self.stats['min_degree']}")
        report.append("")
        
        # 聚类系数
        report.append("【聚类系数】")
        if 'global_clustering_user' in self.stats:
            report.append(f"用户投影网络全局聚类系数: {self.stats['global_clustering_user']:.4f}")
        if 'global_clustering_movie' in self.stats:
            report.append(f"电影投影网络全局聚类系数: {self.stats['global_clustering_movie']:.4f}")
        report.append("")
        
        # 路径长度
        report.append("【路径长度】")
        if self.stats.get('avg_path_length') is not None:
            report.append(f"平均路径长度: {self.stats['avg_path_length']:.4f}")
        if self.stats.get('diameter') is not None:
            report.append(f"网络直径: {self.stats['diameter']}")
        report.append("")
        
        # 度分布
        report.append("【度分布特性】")
        if self.stats.get('power_law_exponent') is not None:
            report.append(f"幂律指数 (γ): {self.stats['power_law_exponent']:.4f}")
            report.append(f"是否为幂律分布: {self.stats['is_power_law']}")
        else:
            report.append("无法确定是否为幂律分布")
        report.append("")
        
        # 度相关性
        report.append("【度相关性】")
        if self.stats.get('degree_correlation') is not None:
            corr = self.stats['degree_correlation']
            if corr > 0:
                report.append(f"度相关性: {corr:.4f} (同配性 - 高度节点倾向于连接高度节点)")
            elif corr < 0:
                report.append(f"度相关性: {corr:.4f} (异配性 - 高度节点倾向于连接低度节点)")
            else:
                report.append(f"度相关性: {corr:.4f} (中性)")
        report.append("")
        
        # 网络类型判断
        report.append("【网络类型判断】")
        avg_degree = self.stats['avg_degree']
        if self.stats.get('global_clustering_user', 0) > 0.1:
            clustering = self.stats.get('global_clustering_user', 0)
            avg_path = self.stats.get('avg_path_length', float('inf'))
            if clustering > 0.1 and avg_path < 10:
                report.append("✓ 具有小世界网络特征（高聚类系数 + 短平均路径长度）")
            else:
                report.append("✗ 不具有明显的小世界网络特征")
        
        if self.stats.get('is_power_law', False):
            report.append("✓ 具有无标度网络特征（度分布遵循幂律分布）")
        else:
            report.append("✗ 不具有明显的无标度网络特征")
        report.append("")
        
        # 关键节点
        report.append("【关键节点】")
        report.append("Top-10 高影响力用户:")
        for i, (node, degree) in enumerate(self.stats.get('top_users', [])[:10], 1):
            user_id = node.replace('U_', '')
            report.append(f"  {i}. 用户 {user_id}: 度 = {degree}")
        
        report.append("")
        report.append("Top-10 高影响力电影:")
        for i, (node, degree) in enumerate(self.stats.get('top_movies', [])[:10], 1):
            movie_id = node.replace('M_', '')
            movie_info = self.movies[self.movies['movieId'] == int(movie_id)]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                report.append(f"  {i}. {title} (ID: {movie_id}): 度 = {degree}")
            else:
                report.append(f"  {i}. 电影 {movie_id}: 度 = {degree}")
        
        report.append("")
        report.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"分析报告已保存到: {save_path}")
        print("\n" + report_text)
    
    def run_full_analysis(self, output_dir='output', skip_clustering=False, fast_mode=False, use_gpu=None, sample_size=2000, no_sampling=False):
        """运行完整分析流程
        
        Args:
            output_dir: 输出目录
            skip_clustering: 是否跳过聚类系数计算（如果太慢）
            fast_mode: 快速模式（跳过聚类系数和路径长度计算）
            use_gpu: 是否使用GPU加速（None表示自动检测）
            sample_size: 采样节点数量（用于聚类系数计算，默认2000）
            no_sampling: 是否禁用采样，强制使用全部节点
        """
        print("=" * 60)
        print("开始网络拓扑特性分析")
        if fast_mode:
            print("【快速模式：将跳过聚类系数和路径长度计算】")
        elif skip_clustering:
            print("【将跳过聚类系数计算】")
        if use_gpu is not None:
            print(f"【GPU模式: {'启用' if use_gpu else '禁用'}】")
        if no_sampling:
            print("【禁用采样模式：将使用全部节点计算】")
        print("=" * 60)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 构建图
        self.build_bipartite_graph()
        
        # 3. 计算各种统计特性
        self.calculate_basic_stats()
        
        if not fast_mode and not skip_clustering:
            # 使用采样方法计算聚类系数（更快），支持GPU加速
            # 如果设置了--no-sampling，则禁用采样
            use_sampling = not no_sampling
            self.calculate_clustering_coefficient(use_sampling=use_sampling, sample_size=sample_size, use_gpu=use_gpu)
        else:
            print("\n=== 跳过聚类系数计算 ===")
            self.stats['global_clustering_user'] = None
            self.stats['global_clustering_movie'] = None
        
        if not fast_mode:
            self.calculate_path_length()
        else:
            print("\n=== 跳过路径长度计算 ===")
            self.stats['avg_path_length'] = None
            self.stats['diameter'] = None
        
        self.analyze_degree_distribution()
        self.calculate_degree_correlation()
        self.identify_key_nodes()
        
        # 4. 可视化
        self.visualize_degree_distribution(save_path=str(output_path / 'network_degree_distribution.png'))
        
        # 5. 生成报告
        self.generate_report(save_path=str(output_path / 'network_analysis_report.txt'))
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        return self.stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='网络拓扑特性分析')
    parser.add_argument('--fast', action='store_true', 
                       help='快速模式：跳过聚类系数和路径长度计算（适合大型网络）')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='跳过聚类系数计算（只跳过聚类系数，仍计算路径长度）')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='输出目录（默认: output）')
    parser.add_argument('--use-gpu', action='store_true',
                       help='使用GPU加速计算（需要安装PyTorch）')
    parser.add_argument('--no-gpu', action='store_true',
                       help='强制使用CPU计算（即使GPU可用）')
    parser.add_argument('--sample-size', type=int, default=2000,
                       help='采样节点数量（用于聚类系数计算，默认2000。越大越准确但越慢）')
    parser.add_argument('--no-sampling', action='store_true',
                       help='禁用采样，强制使用全部节点计算（适合节点数<10000的网络）')
    
    args = parser.parse_args()
    
    # 确定是否使用GPU
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    # 否则自动检测
    
    # 显示GPU状态
    if GPU_AVAILABLE and TORCH_AVAILABLE:
        print("✓ PyTorch GPU已安装且可用")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        except:
            pass
    elif TORCH_AVAILABLE:
        print("⚠ PyTorch已安装，但未检测到GPU设备")
        print("  将使用CPU计算")
    else:
        print("⚠ PyTorch未安装，将使用CPU计算")
        print("  如需GPU加速，请安装: pip install torch")
        print("  (PyTorch会自动检测并使用GPU)")
    
    # 创建分析器并运行完整分析
    analyzer = NetworkAnalyzer()
    stats = analyzer.run_full_analysis(
        output_dir=args.output_dir,
        skip_clustering=args.skip_clustering,
        fast_mode=args.fast,
        use_gpu=use_gpu,
        sample_size=args.sample_size,
        no_sampling=args.no_sampling
    )
    
    print("\n分析结果摘要:")
    print(f"- 节点数: {stats['num_nodes']}")
    print(f"- 边数: {stats['num_edges']}")
    print(f"- 平均度: {stats['avg_degree']:.2f}")
    if stats.get('power_law_exponent'):
        print(f"- 幂律指数: {stats['power_law_exponent']:.2f}")
    if stats.get('global_clustering_user') is not None:
        print(f"- 用户网络聚类系数: {stats['global_clustering_user']:.4f}")
    if stats.get('avg_path_length') is not None:
        print(f"- 平均路径长度: {stats['avg_path_length']:.2f}")

