/*
//普通高斯算法
#include<iostream>
#include<iomanip>
#include<vector>
#include<algorithm>
#include<math.h>
#include<map>
#include<queue>
#include<unordered_map>
typedef long long LL;
using namespace std;
int N;
vector<vector<double> > f(102,vector<double>(102, 0));
double esp = 1e-6;
void input() {
    cin >> N;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N + 1; j++) {
            cin >> f[i][j];
        }
    }
    return;
}
int solve() {
    for (int i = 1; i <= N; i++) {
        int r = i;
        for (int k = i ; k <= N; k++) {
            if (fabs(f[k][i])>esp) { r=k; break; }
        }
        if (r != i)(swap(f[i], f[r]));
        if (fabs(f[i][i]) < esp) { return 0; };
        for (int j = N + 1; j >= i; j--) {
            f[i][j] /= f[i][i];
        }
        for (int k = i + 1; k <= N; k++) {
            for (int j = N + 1; j >= i; j--) {
                f[k][j] -= f[k][i] * f[i][j];
            }
        }
    }
    for (int k = N - 1; k >= 1; k--) {
        for (int j = N; j > k; j--) {
            f[k][N + 1] -= f[j][N + 1]*f[k][j];
        }
    }
    return 1;
}
int main() {
    input();
    if (solve()) {
        for (int i = 1; i <= N; i++) {
            cout << fixed << setprecision(2) << f[i][N + 1] << endl;
        }

    }
    else {
        cout << "No Solution" << endl;
    }
    return 0;
}
*/
/*
//行划分
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <chrono>

void initialize_matrix(int n, double** A, double* b) {
    // Initialize random seed
    srand(time(nullptr));

    // Generate random coefficients for matrix A and vector b
    for (int i = 0; i < n; ++i) {
        A[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand() % 100; // Random coefficient between 0 and 99
        }
        b[i] = rand() % 100; // Random value for vector b
    }
}

void gauss_elimination(int n, double** A, double* b, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        // Broadcast pivot row to all processes
        MPI_Bcast(A[k], n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        // Gaussian elimination
        for (int i = k + 1; i < n; ++i) {
            if (i % size == rank) {
                double factor = A[i][k] / A[k][k];
                for (int j = k; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Back substitution
    for (int k = n - 1; k >= 0; --k) {
        if (k % size == rank) {
            b[k] /= A[k][k];
            A[k][k] = 1.0;
            for (int i = k - 1; i >= 0; --i) {
                if (i % size == rank) {
                    b[i] -= A[i][k] * b[k];
                    A[i][k] = 0.0;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 2000;
    double** A = new double* [n];
    for (int i = 0; i < n; ++i) {
        A[i] = new double[n];
    }
    double b[n];

    // Initialize matrix A and vector b with random values
    if (rank == 0) {
        initialize_matrix(n, A, b);
    }

    // Broadcast matrix A and vector b to all processes
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(A[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform Gaussian elimination
    auto start_time = std::chrono::steady_clock::now();
    gauss_elimination(n, A, b, rank, size);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double elapsed_time = elapsed_seconds.count();

    // Print results and time
    if (rank == 0) {
        std::cout << "Time: " << elapsed_time << " seconds" << std::endl;
    }

    // Clean up
    for (int i = 0; i < n; ++i) {
        delete[] A[i];
    }
    delete[] A;

    MPI_Finalize();
    return 0;
}
*/
/*
//AVX优化
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h> // 包含 SIMD 指令集的头文件

void initialize_matrix(int n, double** A, double* b) {
    // Initialize random seed
    srand(time(nullptr));

    // Generate random coefficients for matrix A and vector b
    for (int i = 0; i < n; ++i) {
        A[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand() % 100; // Random coefficient between 0 and 99
        }
        b[i] = rand() % 100; // Random value for vector b
    }
}

void gauss_elimination(int n, double** A, double* b, int rank, int size) {
    __m256d vec_factor;
    for (int k = 0; k < n; ++k) {
        // Broadcast pivot row to all processes
        MPI_Bcast(A[k], n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        // Gaussian elimination
        for (int i = k + 1; i < n; ++i) {
            if (i % size == rank) {
                double factor = A[i][k] / A[k][k];
                vec_factor = _mm256_set1_pd(factor); // 使用 SIMD 指令设置 factor 的向量
                for (int j = k; j < n; j += 4) { // 每次并行计算 4 个元素
                    __m256d vec_A = _mm256_loadu_pd(&A[k][j]); // 加载 A[k][j] 到向量
                    __m256d vec_b = _mm256_loadu_pd(&A[i][j]); // 加载 A[i][j] 到向量
                    vec_A = _mm256_mul_pd(vec_factor, vec_A); // 向量相乘
                    vec_b = _mm256_sub_pd(vec_b, vec_A); // 向量相减
                    _mm256_storeu_pd(&A[i][j], vec_b); // 存储结果到 A[i][j]
                }
                b[i] -= factor * b[k];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Back substitution
    for (int k = n - 1; k >= 0; --k) {
        if (k % size == rank) {
            b[k] /= A[k][k];
            A[k][k] = 1.0;
            for (int i = k - 1; i >= 0; --i) {
                if (i % size == rank) {
                    b[i] -= A[i][k] * b[k];
                    A[i][k] = 0.0;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 2000;
    double** A = new double* [n];
    for (int i = 0; i < n; ++i) {
        A[i] = new double[n];
    }
    double b[n];

    // Initialize matrix A and vector b with random values
    if (rank == 0) {
        initialize_matrix(n, A, b);
    }

    // Broadcast matrix A and vector b to all processes
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(A[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform Gaussian elimination
    auto start_time = std::chrono::steady_clock::now();
    gauss_elimination(n, A, b, rank, size);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double elapsed_time = elapsed_seconds.count();

    // Print results and time
    if (rank == 0) {
        std::cout << "Time: " << elapsed_time << " seconds" << std::endl;
    }

 
    MPI_Finalize();
    return 0;
}
*/
/*
* 
// MPI特殊高斯消元
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;

const int N = 400;
const int Len = 255;
vector<vector<int>> a(5, vector<int>(90000, 0)); // 消元子
int c[90000];
vector<int> reca(5, 0);

int String_to_int(string a) {
    int res = 0;
    for (char ch : a) {
        res = res * 10 + (ch - '0');
    }
    return res;
}

string int_to_String(int a) {
    ostringstream os;
    os << a;
    return os.str();
}

void input(istringstream& s, vector<int>& q) {
    string st;
    while (s >> st) {
        q[String_to_int(st)] = 1;
    }
}

void inFile(const string& load, const vector<int>& s) {
    ofstream fil(load, ios::app);
    bool flag = false;
    for (int i = Len; i >= 0; i--) {
        if (s[i]) {
            if (!flag) c[i] = 1;
            flag = true;
            fil << int_to_String(i) << " ";
        }
    }
    if (!flag) {
        if (load == "res2.txt") {
            fil << endl;
        }
        return;
    }
    fil << endl;
}

vector<int> xiaoyuan(const vector<int>& s, const vector<int>& q) {
    vector<int> result(s.size(), 0);
    for (size_t i = 0; i < s.size(); i++) {
        result[i] = s[i] ^ q[i];
    }
    return result;
}

void get_duijiaoxian(int s[]) {
    // Function implementation
}

void get_xyz() {
    // Function implementation
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 文件的初始化
    if (rank == 0) {
        ofstream ofs1("res1.txt", ios::trunc);
        ofstream ofs2("res0.txt", ios::trunc);
        ofstream ofs3("res2.txt", ios::trunc);
        ofs1.close();
        ofs2.close();
        ofs3.close();

        string sourceFile = "被消元行.txt";
        string targetFile = "res1.txt";
        ifstream source(sourceFile, ios::binary);
        ofstream target(targetFile, ios::binary);
        if (!source.is_open()) {
            cerr << "Error: Could not open source file " << sourceFile << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!target.is_open()) {
            cerr << "Error: Could not open target file " << targetFile << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        target << source.rdbuf();
        source.close();
        target.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ifstream file("消元子.txt");
    if (!file.is_open()) {
        cerr << "Error: Could not open 消元子.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string line;
    int curfile = 1;
    string curFile = "res" + int_to_String(curfile) + ".txt";
    ofstream fileoutres("res2.txt", ios::app);
    ifstream fileout("res1.txt");
    ifstream fileout1("res0.txt");

    if (!fileout.is_open()) {
        cerr << "Error: Could not open res1.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!fileout1.is_open()) {
        cerr << "Error: Could not open res0.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!fileoutres.is_open()) {
        cerr << "Error: Could not open res2.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bool flagg = true;
    int signal[2] = { 0, 0 };

    while (flagg) {
        ifstream fileout("res1.txt");
        ifstream fileout1("res0.txt");
        int num = 0;
        int num1 = 0;
        flagg = true;
        int needle = 0;
        while (a.size() > 5) {
            a.pop_back();
        }
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j <= Len; j++) {
                a[i][j] = 0;
            }
        }
        while (needle < 5 && getline(file, line)) {
            istringstream stream(line);
            string str;
            int flag = false;
            while (stream >> str) {
                if (!flag) {
                    reca[needle] = String_to_int(str);
                    flag = true;
                }
                a[needle][String_to_int(str)] = 1;
            }
            needle++;
        }

        int p = 0;
        while (p < signal[curfile]) {
            getline(curfile == 1 ? fileout : fileout1, line);
            p++;
        }

        while (getline(curfile == 1 ? fileout : fileout1, line)) {
            signal[curfile]++;
            flagg = false;
            int start = 0;
            istringstream stream(line);
            vector<int> b(90000, 0);
            bool flag = true;
            string str;
            while (stream >> str) {
                if (flag) {
                    start = String_to_int(str);
                    flag = false;
                }
                b[String_to_int(str)] = 1;
            }
            flag = false;
            for (int i = 0; i < a.size(); i++) {
                if (start > reca[i]) {
                    flag = true;
                    a.insert(a.begin() + i, b);
                    reca.insert(reca.begin() + i, start);
                    inFile("res2.txt", b);
                    break;
                }
                else if (start < reca[i]) {
                    continue;
                }
                else if (start == reca[i]) {
                    b = xiaoyuan(b, a[i]);
                    while (start >= 0 && !b[start]) {
                        start--;
                    }
                    if (start < 0) break;
                }
            }
            if (!flag) {
                num1++;
                string curF = "res" + int_to_String(curfile ^ 1) + ".txt";
                inFile(curF, b);
            }
        }

        curfile ^= 1;
        if (flagg) break;
        fileout.close();
        fileout1.close();
        flagg = true;
    }

    fileout.close();
    fileout1.close();
    fileoutres.close();

    MPI_Finalize();
    return 0;
}

*/