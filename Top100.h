#pragma once

#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <queue>
#include <algorithm>	// used by max/min/sort

#include <numeric>		// 引入accumulate 做向量求和 [http://www.cplusplus.com/reference/numeric/accumulate/]


using namespace std;

struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}	
};

struct ListNode {
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
	
};

struct DbListNode {
	pair<int,int> val;
	DbListNode* front;
	DbListNode* next;
	DbListNode() : val(), front(nullptr), next(nullptr) {}
	DbListNode(pair<int, int> x) : val(x), front(nullptr), next(nullptr) {}
	DbListNode(pair<int, int> x, DbListNode* front) : val(x), front(front), next(nullptr) {}
	DbListNode(pair<int, int> x, DbListNode* front, DbListNode* next) : val(x), front(front), next(next) {}
	
};

// 高级数据结构 Tried树
#define TRIE_MAX_CHAR_NUM 26
struct TrieNode {
	TrieNode* child[TRIE_MAX_CHAR_NUM];
	bool is_end;
	string word;
	TrieNode() : is_end(false) {
		for (int i = 0; i < TRIE_MAX_CHAR_NUM; i++) {
			child[i] = 0;
		}
	}
};

// 1. Maximal Rectangle
int maximalRectangle(vector<vector<char>>& matrix);
int largestRectangleArea_N(vector<int>& heights);

// 2.Binary Tree Inorder Traversal (中序遍历)
vector<int> inorderTraversal(TreeNode* root);

// 3. Unique Binary Search Trees
int numTrees(int n);

// 4.Validate Binary Search Tree
bool isValidBST(TreeNode* root);
bool checkBST(TreeNode* root, long long localMin, long long localMax);

// 5. Symmetric Tree
bool isSymmetric(TreeNode* root);
void inOrdTravel(TreeNode* root, vector<int>& nodes, int level);

// 6. Binary Tree Level Order Traversal
vector<vector<int>> levelOrder(TreeNode* root);
void preoderTraversal(TreeNode* root, vector<vector<int>>& result, int level);

// 7. Maximum Depth of Binary Tree
int maxDepth(TreeNode* root);
void preoderTraversalDepth(TreeNode* root, int& maxLevel, int level);

// 8.Construct Binary Tree from Preorder and Inorder Traversal
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);
TreeNode* buildRoot(vector<int>& preorder, vector<int>& inorder, int& pre, int& in, int &end);

// 9. Best Time to Buy and Sell Stock
int maxProfit(vector<int>& prices);
int maxProfet_DP(vector<int>& prices);

// 10. int maxPathSum(TreeNode* root)
int maxPathSum(TreeNode* root);
int checkMax(TreeNode* root);

// 11. Longest Consecutive Sequence
int longestConsecutiveN(vector<int>& nums);
int longestConsecutiveNlogN(vector<int>& nums);

// 12. Single Number
// 异或法！
int singleNumber(vector<int>& nums);

// 13. Word Break
bool wordBreak(string s, vector<string>& wordDict);

// 14. Sort List
ListNode* sortList(ListNode* head);
ListNode* merge(ListNode* list1, ListNode* list2);
ListNode* findMid(ListNode* head);

// 15. Maximum Product Subarray
int maxProduct(vector<int>& nums);

// 16. Majority Element
// 方法一：unordered_map法
int majorityElement(vector<int>& nums);
// 方法二：Boyer–Moore majority vote algorithm 
// 线性时间 常数空间找majority的算法
int majorityElementBoyerMoore(vector<int>& nums);

// 17. Word Search II
vector<string> findWords(vector<vector<char>>& board, vector<string>& words);
void searchBoard(vector<vector<char>>& board, TrieNode* ptr, vector<string> &result, int i, int j);
TrieNode* buildTrie(vector<string>& words);

// 18. Maximal Square
int maximalSquare(vector<vector<char>>& matrix);
int largestSquareArea_N(vector<int>& heights);

// 19. Invert Binary Tree
TreeNode* invertTree(TreeNode* root);

// 20. Kth Smallest Element in a BST
// *返回set最后一个元素：*elements.rbegin();
// *删除set最后一个元素：elements.erase(*elements.rbegin());
int kthSmallest(TreeNode* root, int k);

// 21. Palindrome Linked List
bool isPalindrome(ListNode* head);
ListNode* reverse(ListNode* head);

// 21. Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums);

// 22. Sliding Window Maximum
vector<int> maxSlidingWindow(vector<int>& nums, int k);

// 23. Search a 2D Matrix II
bool searchMatrix(vector<vector<int>>& matrix, int target);

// 24. Perfect Squares
int numSquare_StaticDP(int n);		// 4ms
// dp思想（LeetCode 322 找零钱）
int numSquares(int n);
int coinChange(vector<int> coins, int amount);
// 贪心思想，只适用于特定面值，此题不适用
int checkN(int n, vector<int> candidates, int& result);

// 25. Move Zeroes
void moveZeroes(vector<int>& nums);

// 26. Find the Duplicate Number
// Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
// 利用快慢指针
int findDuplicate(vector<int>& nums);

// 27.  Best Time to Buy and Sell Stock with Cooldown
int maxProfit_withColldown(vector<int>& prices);

// 28. House Robber III
/*
	 1. All houses in this place forms a binary tree.
	 2. Automatically contact the police if two directly-linked houses were broken into on the same night.
*/
int rob(TreeNode* root);
vector<int> rob_planner(TreeNode* root);

// 29. Counting Bits
// Given a non negative integer number num.\
// For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.
vector<int> countBits(int num);

// 30. Top K Frequent Elements
// Quickselect: a textbook algorthm typically used to solve the problems "find kth something": kth smallest, kth largest, kth most frequent, kth less frequent, etc. 
vector<int> topKFrequent(vector<int>& nums, int k);

// 31.  Decode String
string decodeString(string s);
string decodeString(string s, int& i);

// 32. Queue Reconstruction by Height
// sort 及 insert 的高级使用方法
vector<vector<int>> reconstructQueue(vector<vector<int>>& people);

// 33. Partition Equal Subset Sum
bool canPartition(vector<int>& nums);

// 34. Path Sum III
// You are given a binary tree in which each node contains an integer value.
// Find the number of paths that sum to a given value.
int pathSum(TreeNode* root, int sum);
void pathCounter(TreeNode* root, int curSum, int target, unordered_map<int, int>& preSum, int& count);

// 35. Find All Anagrams in a String
// Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
vector<int> findAnagrams(string s, string p);

// 36. Target Sum
int findTargetSumWays(vector<int>& nums, int S);

/*
*	let's start with int[][] dp = new int[nums.length][s + 1] where dp is 2-d array with dp[i][j] means number of ways to get sum j with first i elements from nums.
*	Then you have the transition dp[i][j] = dp[i-1][j] + dp[i][j-nums[i]], 
*	i.e. you can get the sum j either by just repeating all the ways to get sum j by using first i-1 elements, 
*	or add nums[i] on top of each way to get sum j-nums[i] using first i elements
*/
int subsetSum(vector<int>& nums, const int target);

// 37. Subarray Sum Equals K
// Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.
/*
 * if the cumulative sum upto two indices, say i and j is at a difference of kk i.e. if sum[i] - sum[j] = k,
 * the sum of elements lying between indices i and j is k.
 */
int subarraySum(vector<int>& nums, int k);

// 38. Shortest Unsorted Continuous Subarray
/*
* Given an integer array nums, you need to find one continuous subarray that if you only sort this subarray in ascending order, 
* then the whole array will be sorted in ascending order.
* Return the shortest such subarray and output its length.
*/
int findUnsortedSubarray(vector<int>& nums);

// 39. Task Scheduler
int leastInterval(vector<char>& tasks, int n);

// // 40. Palindromic Substrings
// Given a string, your task is to count how many palindromic substrings in this string.
// The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.
// 回文：中心开花法
int countSubstrings(string s);

// 41. Daily Temperatures
// 求右侧i更大的数的个数
vector<int> dailyTemperatures(vector<int>& T);

// 42. Partition Labels
// A string S of lowercase English letters is given. 
// We want to partition this string into as many parts as possible so that each letter appears in at most one part, 
// and return a list of integers representing the size of these parts.
vector<int> partitionLabels(string S);

// 43. Valid Sudoku
bool isValidSudoku(vector<vector<char>>& board);
bool isValidNum(vector<vector<char>>& board, int row, int col, char c);

// 44. Multiply Strings
string multiply(string num1, string num2);

// 45. Wildcard Matching
// Input: s = "aab", p = "c*a*b"		Output: false
bool isMatchWildcard(string s, string p);

// 46. Regular Expression Matching
// Input: s = "aab", p = "c*a*b"		Output: true
bool isMatchRegular(string s, string p);

// 47. Permutations II
// Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.
// *回溯法
vector<vector<int>> permuteUnique_Backtracking(vector<int>& nums);
void DFS_permute(vector<int>& nums, vector<bool>& isUsed, vector<int> item, vector<vector<int>>& result);
// 交换法
vector<vector<int>> permuteUnique_Swap(vector<int>& nums);
void Swap_permute(int begin, vector<int> nums, vector<vector<int>>& result);

// 48. Pow(x,n)——分治法
// Implement pow(x, n), which calculates x raised to the power n (i.e. xn).
double myPow(double x, int n);

// 49. N皇后
vector<vector<string>> solveNQueens(int n);
void nQueensSolver(int row, int n, vector<string>& item, vector<vector<string>>& result);
bool isValidQueen(vector<string> item, int row, int col, int n);

// 50. Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix);

// 51. Add Strings
string addStrings(string num1, string num2);

// 52. LRU Cache
void LRUCache(int capacity);
int get(int key);
void put(int key, int value);