#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

using namespace std;

const double eps = 1e-9;

int d;
vector<vector<double>> p;

double dot_product(const vector<double>& a, const vector<double>& b) {
    double res = 0;
    for (int i = 0; i < d; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

vector<double> cross_product(const vector<double>& a, const vector<double>& b) {
    vector<double> res(d);
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
    return res;
}

double det(const vector<vector<double>>& m) {
    int d = m.size();
    if (d == 1) {
        return m[0][0];
    }
    double res = 0;
    for (int i = 0; i < d; ++i) {
        vector<vector<double>> sub_m(d - 1, vector<double>(d - 1));
        for (int j = 1; j < d; ++j) {
            for (int k = 0; k < d; ++k) {
                if (k == i) continue;
                sub_m[j - 1][k < i ? k : k - 1] = m[j][k];
            }
        }
        res += (i % 2 == 0 ? 1 : -1) * m[0][i] * det(sub_m);
    }
    return res;
}


vector<double> solve_linear_system(const vector<vector<double>>& A, const vector<double>& b) {
    vector<vector<double>> m(d, vector<double>(d + 1));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            m[i][j] = A[i][j];
        }
        m[i][d] = b[i];
    }

    for (int col = 0; col < d; ++col) {
        int row_with_max_value = col;
        for (int row = col + 1; row < d; ++row) {
            if (abs(m[row][col]) > abs(m[row_with_max_value][col])) {
                row_with_max_value = row;
            }
        }
        swap(m[col], m[row_with_max_value]);

        for (int row = col + 1; row < d; ++row) {
            double c = m[row][col] / m[col][col];
            for (int j = col; j <= d; ++j) {
                m[row][j] -= c * m[col][j];
            }
        }
    }

    vector<double> x(d);
    for (int row = d - 1; row >= 0; --row) {
        x[row] = m[row][d];
        for (int col = row + 1; col < d; ++col) {
            x[row] -= x[col] * m[row][col];
        }
        x[row] /= m[row][row];
    }

    return x;
}

bool is_inside_simplex(const vector<vector<double>>& simplex, const vector<double>& point) {
    vector<vector<double>> A(d, vector<double>(d));
    vector<double> b(d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            A[i][j] = simplex[j + 1][i] - simplex[0][i];
        }
        b[i] = point[i] - simplex[0][i];
    }

    auto coefs = solve_linear_system(A, b);

    double sum_coefs = accumulate(coefs.begin(), coefs.end(), 0.);

    return all_of(coefs.begin(), coefs.end(), [](double coef) 
    {
            return coef >= -eps; }) && sum_coefs <= 1 + eps;
    }

vector<vector<double>> find_convex_hull() {
    vector<vector<double>> res;
    while (p.size() >= d + 1) {
        mt19937 gen(random_device{}());
        uniform_real_distribution<> dis(-1, 1);

        vector<vector<double>> simplex(d + 1);
        bool found = false;
        while (!found) {
            vector<vector<double>> dirs(d + 1, vector<double>(d));
            for (int i = 0; i < d + 1; ++i) {
                for (int j = 0; j < d; ++j) {
                    dirs[i][j] = dis(gen);
                }
            }

            for (int i = 0; i < d + 1; ++i) {
                int max_point_index = -1;
                double max_dot_product = -1e18;
                for (int j = 0; j < p.size(); ++j) {
                    double dp = dot_product(dirs[i], p[j]);
                    if (dp > max_dot_product) {
                        max_dot_product = dp;
                        max_point_index = j;
                    }
                }
                simplex[i] = p[max_point_index];
            }

            vector<vector<double>> m(d, vector<double>(d));
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    m[i][j] = simplex[j + 1][i] - simplex[0][i];
                }
            }

            if (abs(det(m)) > eps) {
                found = true;
            }
        }

        for (const auto& point : simplex) {
            res.push_back(point);
        }

        p.erase(remove_if(p.begin(), p.end(), [&](const vector<double>& point) {
            return is_inside_simplex(simplex, point);
        }), p.end());
    }

    return res;
}

int main() {
    cin >> d; 

    int n;
    cin >> n;

    p.resize(n, vector<double>(d));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            cin >> p[i][j];
        }
    }

    auto convex_hull = find_convex_hull();

    cout << convex_hull.size() << endl;
    for (const auto& point : convex_hull) {
        for (double coord : point) {
            cout << coord << ' ';
        }
        cout << endl;
    }

    return 0;
}