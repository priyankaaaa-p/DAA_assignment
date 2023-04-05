#include <iostream>
#include <cstring>

using namespace std;

const int MAXN = 1005;
const int INF = 1e9;
int dp[MAXN][MAXN];

int main() {
    string s1, s2;
    cin >> s1 >> s2;
    int n = s1.size(), m = s2.size();

    memset(dp, 0, sizeof(dp)); // initialize with 0

    // fill the first row and column
    for (int i = 1; i <= n; i++) {
        dp[i][0] = i;
    }
    for (int j = 1; j <= m; j++) {
        dp[0][j] = j;
    }

    // fill the rest of the matrix
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1]; // no cost
            } else {
                dp[i][j] = min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1])) + 1; // cost of 1
            }
        }
    }

    // print the matrix
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}

dp[i][j] = min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1])) + (s1[i-1] != s2[j-1]);

string align1 = "";
string align2 = "";

int i = n, j = m;

while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] && s1[i-1] == s2[j-1]) {
        // match
        align1 = s1[i-1] + align1;
        align2 = s2[j-1] + align2;
        i--;
        j--;
    } else if (i > 0 && dp[i][j] == dp[i-1][j] + 1) {
        // gap in s2
        align1 = s1[i-1] + align1;
        align2 = "-" + align2;
        i--;
    } else {
        // gap in s1
        align1 = "-" + align1;
        align2 = s2[j-1] + align2;
        j--;
    }
}

