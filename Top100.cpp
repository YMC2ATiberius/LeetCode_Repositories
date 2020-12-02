#include <Top100.h>

using namespace std;
/*
 *	�½�������TreeNode a(23);  ���ּ�Ϊa
 *	�½�ָ�������TreeNode* root = new TreeNode(23);
*/


// 1. Maximal Rectangle
int maximalRectangle(vector<vector<char>>& matrix) {
	if (matrix.empty()) {
		return 0;
	}
	vector<int> heights(matrix[0].size(), 0);
	int result = 0;
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			// �ײ����0��������
			heights[j] += matrix[i][j] == '1' ? 1: -heights[j];
		}
		result = max(result, largestRectangleArea_N(heights));
	}
	return result;
}

// 2.Binary Tree Inorder Traversal (�������)
vector<int> inorderTraversal(TreeNode* root) {
	vector<int> result;
	while (root) {
		if (root->left) {
			TreeNode* temp = root->left;
			while (temp->right && temp->right != root) {
				temp = temp->right;
			}
			if (!temp->right) {
				temp->right = root;
				root = root->left;
			}
			else {
				result.push_back(root->val);
				// ending traversal(6-3-**)
				temp->right = nullptr;
				root = root->right;
			}
		}
		else {
			result.push_back(root->val);
			root = root->right;
		}
	}
	return result;
}

// 3. Unique Binary Search Trees
int numTrees(int n) {
	vector<int> results(n + 1, 0);
	results[0] = results[1] = 1;
	for (int i = 2; i <= n; i++) {
		for (int j = 0; j <= i - 1; j++) {
			results[i] += results[j] * results[i - j - 1];
		}
	}
	return results[n];
}

// 4.Validate Binary Search Tree
bool isValidBST(TreeNode* root) {
	return checkBST(root, LLONG_MIN, LLONG_MAX);	
}

// pre-order traversal ��ǰ��
bool checkBST(TreeNode* root, long long localMin, long long localMax) {
	if (root == nullptr) {
		return true;
	}
	if (root->val >= localMax || root->val <= localMin) return false;
	return checkBST(root->left, localMin, root->val) && checkBST(root->right, root->val, localMax);
}

// 5. Symmetric Tree
bool isSymmetric(TreeNode* root) {
	if (root == nullptr) {
		return true;
	}
	vector<int> nodes;
	inOrdTravel(root, nodes, 1);
	if (nodes.size() % 2 == 0) {
		return false;
	}
	int i = 0, j = nodes.size() - 1;
	while (i < j) {
		if (nodes[i] != nodes[j]) return false;
		i++;
		j--;
	}
	return true;
}

void inOrdTravel(TreeNode* root, vector<int>& nodes, int level) {
	if (root == nullptr) {
		return;
	}
	inOrdTravel(root->left, nodes, level + 1);
	nodes.push_back(root->val + level);
	inOrdTravel(root->right, nodes, level + 1);
}

// 6. Binary Tree Level Order Traversal
vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> result;
	preoderTraversal(root, result, 0);
	return result;
}

void preoderTraversal(TreeNode* root, vector<vector<int>>& result, int level) {
	if (root == nullptr) {
		return;
	}
	if (level + 1 > result.size()) {
		vector<int> item(1, root->val);
		result.push_back(item);
	}
	else {
		result[level].push_back(root->val);
	}	
	preoderTraversal(root->left, result, level + 1);
	preoderTraversal(root->right, result, level + 1);
}

// 7. Maximum Depth of Binary Tree
int maxDepth(TreeNode* root) {
	if (root == nullptr) {
		return 0;
	}
	int maxLevel = 1;
	preoderTraversalDepth(root, maxLevel, 1);
	return maxLevel;
}

void preoderTraversalDepth(TreeNode* root, int &maxLevel, int level) {
	if (root == nullptr) {
		return;
	}
	if (level > maxLevel) {
		maxLevel = level;
	}
	preoderTraversalDepth(root->left, maxLevel, level + 1);
	preoderTraversalDepth(root->right, maxLevel, level + 1);
}

// 8.Construct Binary Tree from Preorder and Inorder Traversal
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
	int pre = 0, in = 0, end = INT_MAX;
	return buildRoot(preorder, inorder, pre, in, end);
}

TreeNode* buildRoot(vector<int>& preorder, vector<int>& inorder, int &pre, int &in, int &end) {
	if (pre == preorder.size()) {
		return nullptr;
	}
	if (inorder[in] == end) {
		in++;											// in����ָ���/Ҷ�ڵ�ʱ����
		return nullptr;
	}
	
	int rootVal = preorder[pre++];						// preÿ�ζ�����
	TreeNode* root = new TreeNode(rootVal);				// new���ص���ָ�����������
	root->left = buildRoot(preorder, inorder, pre, in, rootVal);
	root->right = buildRoot(preorder, inorder, pre, in, end);
	return root;	
}

// 9. Best Time to Buy and Sell Stock
int maxProfit(vector<int>& prices) {
	int maxProfit = 0;
	if (prices.empty()) {
		return maxProfit;
	}

	int buy = prices[0];
	for (int i = 1; i < prices.size(); i++) {
		if (prices[i] < buy) {
			buy = prices[i];
			continue;
		}
		maxProfit = max(maxProfit, prices[i] - buy);
	}
	return maxProfit;
}

int maxProfet_DP(vector<int>& prices) {
	int maxCur = 0;		// maxCur = current maximum value
	int maxSoFar = 0;	// maxSoFar = maximum value found so far	
	for (int i = 1; i < prices.size(); i++) {
		maxCur = max(0, maxCur += prices[i] - prices[i - 1]);
		maxSoFar = max(maxCur, maxSoFar);
	}
	return maxSoFar;
}

// 10. int maxPathSum(TreeNode* root)
// ��Ҷ�ӵ�����
// �ڵݹ�������ȫ�ֱ��������������·����ֵ!
int maxVal = INT_MIN;
int maxPathSum(TreeNode* root) {
	checkMax(root);
	return maxVal;
}

int checkMax(TreeNode* root) {
	if (root == nullptr) return 0;
	int leftVal = max(0, checkMax(root->left));
	int rightVal = max(0, checkMax(root->right));
	// �������ڵ�����·��ֵ
	maxVal = max(maxVal, root->val + leftVal + rightVal);
	// ֻ������/�ҽڵ�����·��ֵ
	return root->val + max(leftVal, rightVal);
}

// 11. Longest Consecutive Sequence
// ��һ HashSet
int longestConsecutiveN(vector<int>& nums) {
	if (nums.empty()) return 0;
	set<int> HashSet(nums.begin(), nums.end());		//���д����
	int maxLength = 1, length = 1;
	set<int>::iterator it = HashSet.begin();
	int front = *it;
	for (it++; it != HashSet.end(); it++) {
		int back = *it;
		if (back - front == 1) {
			length++;			
		}
		else {
			length = 1;
		}
		front = *it;
		maxLength = max(length, maxLength);
	}
	return maxLength;
}

// ���� ������ O(NlogN)
int longestConsecutiveNlogN(vector<int>& nums) {
	if (nums.empty()) return 0;
	std::sort(nums.begin(), nums.end());
	int i = 0, j = 1;
	int maxLength = 1;
	while (i < nums.size() && j < nums.size()) {		
		int dis = nums[j] - nums[j - 1];
		if (dis == 1) {
			maxLength = max(maxLength, nums[j] - nums[i] + 1);
		}
		else if (dis > 1) {
			i = j;
		}
		j++;
	}
	return maxLength;
}

// 12. Single Number
// ��򷨣�
int singleNumber(vector<int>& nums) {
	int result = 0;
	for (int i = 0; i < nums.size(); i++) {
		result ^= nums[i];
	}
	return result;
}

// 13. Word Break
// DP
bool wordBreak(string s, vector<string>& wordDict) {
	unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
	vector<bool> result(s.length() + 1, false);

	result[0] = true;
	for (int j = 1; j <= s.length(); j++) {
		for (int i = 0; i < j; i++) {
			string word = s.substr(i, j - i);
			if (result[i] && wordSet.find(word) != wordSet.end()) {
				result[j] = true;
				break;
			}
		}
	}
	return result[s.length()];
}
// ��
bool wordBreak_N2(string s, vector<string>& wordDict) {
	set<string> wordSet(wordDict.begin(), wordDict.end());	
	int i = 0, j = 1;
	vector<pair<int, int>> iVec;
	string word;

	do {
		word = s.substr(i, j - i);
		// ����ҵ�����
		if (wordSet.find(word) != wordSet.end()) {
			iVec.push_back(make_pair(i,j));
			i = j;
			if (j == s.length()) {
				return true;
			}
		}
		else if (j == s.length()) {
			if (iVec.empty()) {
				return false;
			}
			i = iVec.back().first;
			j = iVec.back().second + 1;
			iVec.pop_back();
			continue;
		}
		j++;
	} while (i < s.length() && j <= s.length());
	return false;
}

// 14. Sort List
ListNode* sortList(ListNode* head) {
	if (!head || !head->next) {
		return head;
	}
	ListNode* mid = findMid(head);
	ListNode* list1 = sortList(head);
	ListNode* list2 = sortList(mid);
	return merge(list1, list2);
}

ListNode* merge(ListNode* list1, ListNode* list2) {
	ListNode newHead(0);
	ListNode* ptr = &newHead;
	while (list1 && list2) {
		if (list1->val < list2->val) {
			ptr->next = list1;
			list1 = list1->next;			
		}
		else {
			ptr->next = list2;
			list2 = list2->next;
		}
		ptr = ptr->next;
	}
	if (list1) {
		ptr->next = list1;
	}
	else {
		ptr->next = list2;
	}
	return newHead.next;
}

// ��2���ٿ�ָ�뵽���βʱ����ָ�����õ�mid��
ListNode* findMid(ListNode* head) {
	ListNode* midPre = nullptr;
	while (head && head->next) {
		if (!midPre) {
			midPre = head;
		}
		else {
			midPre = midPre->next;
		}		
		head = head->next->next;
	}
	ListNode* mid = head;
	if (midPre) {
		mid = midPre->next;
	}	
	midPre->next = nullptr;
	return mid;
}

//// �����ָ��
//ListNode* nextSubList = new ListNode();											// �ڶ��������ʹ�õı����ɶ���Ϊȫ�ֱ��������������õ�ʹ�ã�
//ListNode* split(ListNode* start, int size) {
//	ListNode* midPrev = start;
//	ListNode* end = start->next;
//	//use fast and slow approach to find middle and end of second linked list
//	for (int index = 1; index < size && (midPrev->next || end->next); index++) {
//		if (end->next) {
//			end = (end->next->next) ? end->next->next : end->next;
//		}
//		if (midPrev->next) {
//			midPrev = midPrev->next;
//		}
//	}
//	ListNode* mid = midPrev->next;
//	nextSubList = end->next;
//	midPrev->next = nullptr;
//	end->next = nullptr;
//	// return the start of second linked list
//	return mid;
//}

// 15. Maximum Product Subarray
int maxProduct(vector<int>& nums) {
	int result = nums[0];
	int n = nums.size();
	// int imin = result, imax = result;
	// ��ʡ�ڴ� дѭ������
	for (int i = 1, imin = result, imax = result; i < n; i++) {
		/* imax/imin stores the max/min product of
		 * subarray that ends with the current number A[i].
		 * multiplied by a negative makes big number smaller, small number bigger
		 so we redefine the extremums by swapping them.
		 */
		if (nums[i] < 0) {
			swap(imin, imax);
		}
		// max/min product for the current number is either the current number itself
		// or the max/min by the previous number times the current one
		imax = max(nums[i], imax * nums[i]);
		imin = min(nums[i], imin * nums[i]);
		result = max(result, imax);
	}
	return result;
}

// 16. Majority Element
// ����һ��unordered_map��
int majorityElement(vector<int>& nums) {
	int len = nums.size();
	if (len == 1) {
		return nums[0];
	}
	int result = 0;
	unordered_map<int, int> cont;
	for (int number : nums) {
		if (cont.find(number) == cont.end()) {
			cont[number] = 1;
		}
		else {
			cont[number]++;
			if (cont[number] > len / 2) {
				result = number;
				break;
			}
		}
	}
	return result;
}

// ��������Boyer�CMoore majority vote algorithm 
// ����ʱ�� �����ռ���majority���㷨
int majorityElementBoyerMoore(vector<int>& nums) {
	int result = nums[0], count = 1;
	for (int i = 1; i < nums.size(); i++) {
		if (count == 0) {
			result = nums[i];
			count++;
		}
		else if (nums[i] == result) {
			count++;
		}
		else {
			count--;
		}
	}
	return result;
}

// 17. Word Search II
vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
	vector<string> result;
	TrieNode* root = buildTrie(words);
	/*searchBoard(board, root, result, 0, 0);*/
	for (int i = 0; i < board.size(); i++) {
		for (int j = 0; j < board[0].size(); j++) {
			searchBoard(board, root, result, i, j);
		}
	}
	return result;
}

void searchBoard(vector<vector<char>>& board, TrieNode* ptr, vector<string> &result, int i, int j) {
	char c = board[i][j];
	if (c == '#' || ptr->child[c - 'a'] == nullptr) return;
	ptr = ptr->child[c - 'a'];
	if (ptr->is_end) {
		result.push_back(ptr->word);
		ptr->is_end = false;		// de-duplicate?
	}

	// DFS ���淽������
	// ��'#'����mark����ѷ��ʣ�
	// ע�� board.size() - 1
	board[i][j] = '#';
	if (i > 0) searchBoard(board, ptr, result, i - 1, j);
	if (j > 0) searchBoard(board, ptr, result, i, j - 1);
	if (i < board.size() - 1) searchBoard(board, ptr, result, i + 1, j);
	if (j < board[0].size() - 1)searchBoard(board, ptr, result, i, j + 1);
	board[i][j] = c;
}

TrieNode* buildTrie(vector<string>& words) {
	TrieNode* root = new TrieNode();
	for (string word : words) {
		TrieNode* ptr = root;
		for (int i = 0; i < word.size(); i++) {
			int pos = word[i] - 'a';
			if (ptr->child[pos] == nullptr) {
				ptr->child[pos] = new TrieNode();
			}
			ptr = ptr->child[pos];
		}
		ptr->is_end = true;
		ptr->word = word;
	}
	return root;
}

// 18. Maximal Square
int maximalSquare(vector<vector<char>>& matrix) {
	if (matrix.empty()) {
		return 0;
	}
	vector<int> heights(matrix[0].size(), 0);
	int result = 0;
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			// �ײ����0��������
			heights[j] += matrix[i][j] == '1' ? 1 : -heights[j];
		}
		result = max(result, largestSquareArea_N(heights));
	}
	return result;
}

int largestSquareArea_N(vector<int>& heights) {
	if (heights.empty()) {
		return 0;
	}

	vector<int> leftLessMin(heights.size(), -1);
	// left ��������
	for (int i = 1; i < heights.size(); i++) {
		int left = i - 1;
		while (left > -1 && heights[left] >= heights[i]) {
			left = leftLessMin[left];
		}
		// �����ֵ
		leftLessMin[i] = left;
	}
	vector<int> rightLessMin(heights.size(), heights.size());
	// right ��������
	for (int i = heights.size() - 2; i > -1; i--) {
		int right = i + 1;
		while (right < heights.size() && heights[right] >= heights[i]) {
			right = rightLessMin[right];
		}
		rightLessMin[i] = right;
	}

	int result = 0;
	for (int i = 0; i < heights.size(); i++) {
		int side = min(rightLessMin[i] - leftLessMin[i] - 1, heights[i]);
		result = max(result, side * side);
	}
	return result;
}

// 19. Invert Binary Tree
TreeNode* invertTree(TreeNode* root) {
	if (root == nullptr)	return root;
	queue<TreeNode*> Q;
	Q.push(root);

	while (!Q.empty()) {
		TreeNode* node = Q.front();
		Q.pop();

		
		if (node->left != nullptr) {
			Q.push(node->left);
		}
		if (node->right != nullptr) {
			Q.push(node->right);
		}

		TreeNode* temp = node->left;
		node->left = node->right;
		node->right = temp;
	}
	return root;
}

// 20. Kth Smallest Element in a BST
int kthSmallest(TreeNode* root, int k) {
	queue<TreeNode*> Q;
	set<int> elements;
	Q.push(root);

	while (!Q.empty()) {
		TreeNode* node = Q.front();
		Q.pop();
		int num = node->val;

		if (elements.size() < k) {
			elements.insert(num);
		}
		else {
			if (num < *elements.rbegin()) {
				elements.erase(*elements.rbegin());
				elements.insert(num);
			}			
		}

		if (node->left != nullptr) {
			Q.push(node->left);
		}
		if (node->right != nullptr) {
			Q.push(node->right);
		}
	}
	return *elements.rbegin();
}

// 21. Palindrome Linked List
// ���ε�reverse����ǰ��Σ���
bool isPalindrome(ListNode* head) {
	if (head == nullptr) return true;
	ListNode *fast = head, *slow = head;
	while (fast != nullptr && fast->next != nullptr)
	{
		fast = fast->next->next;
		slow = slow->next;
	}
	// ������ slow������һλ �����Ƚ��м���
	if (fast != nullptr) {
		slow = slow->next;
	}
	slow = reverse(slow);
	fast = head;

	while (slow != nullptr) {
		if (slow->val != fast->val)
			return false;
		slow = slow->next;
		fast = fast->next;
	}
	return true;
}

ListNode* reverse(ListNode* head) {
	ListNode* front = new ListNode(0);
	while (head != nullptr) {
		ListNode *next = head->next;
		head->next = front->next;
		front->next = head;
		head = next;
	}
	return front->next;
}

// 21. Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums) {
	int N = nums.size();
	vector<int> product(N, 1);		// 1.leftProduct => product

	for (int i = 1; i < N; i++) {
		product[i] = product[i - 1] * nums[i - 1];
	}
	int right = 1;
	for (int i = N - 2; i > -1; i--) {
		right *= nums[i + 1];
		product[i] *= right;
	}
	return product;
}

// 22. Sliding Window Maximum
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	vector<int> result;
	deque<int> window;
	int N = nums.size();
	for (int i = 0; i < N; i++) {
		// С����߽�
		if (!window.empty() && window.front() < i - k + 1)
		{
			window.pop_front();
		}
		// ��βԪ�ر���Ԫ�ػ�С���򲻿��ܳ�Ϊ���Ԫ��
		while (!window.empty() && nums[window.back()] < nums[i])
		{
			window.pop_back();
		}
		window.push_back(i);
		// �����ڷ�������k(0~k-1)ʱ��ʼ¼����
		if (i >= k - 1) {
			result.push_back(nums[window.front()]);
		}			
	}
	return result;
}

// 23. Search a 2D Matrix II
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	if (matrix.empty()) return false;
	int length = matrix[0].size();
	for (int i = 0; i < matrix.size(); i++) {		
		if (target > matrix[i][length - 1]) continue;
		int inLeft = 0, inRight = length - 1;
		while (inLeft < inRight) {
			int mid = (inLeft + inRight) / 2;
			if (target == matrix[i][mid]) {
				return true;
			}
			else if (target < matrix[i][mid]) {
				inRight = mid;
			}
			else {
				inLeft = mid + 1;
			}
		}
	}
	return false;
}

// 24. Perfect Squares
int numSquare_StaticDP(int n){
	if (n <= 0) return 0;
		// cntPerfectSquares[i] = the least number of perfect square numbers 
		// which sum to i. Since cntPerfectSquares is a static vector, if 
		// cntPerfectSquares.size() > n, we have already calculated the result 
		// during previous function calls and we can just return the result now.
	static vector<int> allSquares({ 0 });

	while (allSquares.size() <= n) {
		// pushback��num�Զ�����
		int num = allSquares.size();
		int square = INT_MAX;
		// �������п��ܵ�i������ʱ��ǰnum ����Ҫ����candidates
		for (int i = 1; i * i <= num; i++) {
			square = min(square, allSquares[num - i * i] + 1);
		}
		allSquares.push_back(square);
	}
	return allSquares[n];
}

// ����coinChange���ȼ���candidates���ڼ������ٳ�Ʊ��
int numSquares(int n) {
	vector<int> candidates;
	vector<int> dp(n + 1, -1);
	dp[0] = 0;
	for (int i = 1; i <= n; i++) {
		if (i * i <= n) {
			candidates.push_back(i * i);
		}
		for (int j = 0; j < candidates.size(); j++) {
			if (i - candidates[j] >= 0 && dp[i - candidates[j]] != -1) {
				if (dp[i] == -1 || dp[i] > dp[i - candidates[j]] + 1) {
					dp[i] = dp[i - candidates[j]] + 1;
				}
			}
		}
	}	
	return dp[n];
}

// ̰��˼�룬ֻ�������ض���ֵ�����ⲻ����
int checkN(int n, vector<int> candidates, int& result) {
	if (n <= 0) return result;
	for (int i = candidates.size() - 1; i >= 0; i--) {
		if (candidates[i] <= n) {
			result++;
			return checkN(n - candidates[i], candidates, result);			
		}
	}
}

int coinChange(vector<int> coins, int amount) {
	vector<int> dp(amount + 1, -1);
	dp[0] = 0;
	for (int i = 1; i <= amount; i++) {
		for (int j = 0; j < coins.size(); j++) {
			// ��������ֵ��֮ǰ�Ľ���ѱ�dp��
			if (i - coins[j] >= 0 && dp[i - coins[j]] != -1) {
				// δ��ֵ����ֵ��С���Ÿ���
				if (dp[i] == -1 || dp[i] > dp[i - coins[j]] + 1) {
					dp[i] = dp[i - coins[j]] + 1;
				}
			}
		}
	}
	return dp[amount];
}

// 25. Move Zeroes
// ����һ�����롾��ǰ�ġ��������Ե�
void moveZeroes(vector<int>& nums) {
	for (int firstZero = 0, i = 0; i < nums.size(); i++) {
		if (nums[i] != 0) {
			swap(nums[firstZero], nums[i]);
			firstZero++;
		}
	}
}

// 26. Find the Duplicate Number
// Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
// ���ÿ���ָ��
int findDuplicate(vector<int>& nums) {
	if (nums.size() > 1) {
		int slow = nums[0];
		int fast = nums[nums[0]];
		while (slow != fast) {
			slow = nums[slow];
			fast = nums[nums[fast]];
		}

		fast = 0;
		while (slow != fast) {
			slow = nums[slow];
			fast = nums[fast];
		}
		return slow;
	}
	return -1;
}

// ͨ�÷���
int findDuplicate_Universal(vector<int>& nums) {
	unordered_map<int, int> map;
	for (int i = 0; i < nums.size(); i++) {
		if (map.find(nums[i]) != map.end()) {
			return nums[i];
		}
		map[nums[i]] = 1;
	}
	return 1;
}

// 27.  Best Time to Buy and Sell Stock with Cooldown
/*
	buy[i]  = max(rest[i-1]-price, buy[i-1])
	sell[i] = max(buy[i-1]+price, sell[i-1])
	rest[i] = max(sell[i-1], buy[i-1], rest[i-1])
	====
	Well, the answer lies within the fact that buy[i] <= rest[i] which means rest[i] = max(sell[i-1], rest[i-1]). That made sure [buy, rest, buy] is never occurred.

	A further observation is that and rest[i] <= sell[i] is also true therefore

	rest[i] = sell[i-1]
	===
	buy[i] = max(sell[i-2]-price, buy[i-1])
	sell[i] = max(buy[i-1]+price, sell[i-1])
*/
int maxProfit_withColldown(vector<int>& prices) {
	int buy = INT_MIN, sell = 0, pre_buy, pre_sell = 0;
	for (int price : prices) {
		pre_buy = buy;
		buy = max(pre_sell - price, pre_buy);
		pre_sell = sell;
		sell = max(pre_buy + price, pre_sell);
	}
	return sell;
}

// 28. House Robber III
/*
	 1. All houses in this place forms a binary tree.
	 2. Automatically contact the police if two directly-linked houses were broken into on the same night.
*/
int rob(TreeNode* root) {
	vector<int> profits = rob_planner(root);
	return max(profits[0], profits[1]);
}

vector<int> rob_planner(TreeNode* root) {
	if (root == nullptr) {
		return { 0, 0 };
	}
	vector<int> left = rob_planner(root->left);
	vector<int> right = rob_planner(root->right);

	int notRobProfit = max(left[0], left[1]) + max(right[0], right[1]);
	int robProfit = root->val + left[0] + right[0];
	// {��������, ���ٻ���}
	return { notRobProfit, robProfit };
}

// 29. Counting Bits
// Given a non negative integer number num.\
// For every numbers i in the range 0 �� i �� num calculate the number of 1's in their binary representation and return them as an array.
vector<int> countBits(int num) {
	vector<int> result(num + 1);
	for (int i = 1, k = 0; i <= num; i++) {		
		if (i == 1 << (k + 1))	k++;
		int shift = 1 << k;					// shift ��ʾ�ƶ�λ������2~k�η�
		result[i] = 1 + result[i - shift];
	}
	return result;
}

// 30. Top K Frequent Elements
vector<int> topKFrequent(vector<int>& nums, int k) {	
	vector<int> result;
	/*unordered_map<int, int> cont;
	for (int number : nums) {
		if (cont.find(number) == cont.end()) {
			cont[number] = 1;
		}
		else {
			cont[number]++;			
		}
	}

	unordered_map<int, int>::iterator iter;
	list<int> temp;
	for (iter = cont.begin(); iter != cont.end(); iter++) {
		if (temp.size() == 0) {
			temp.push_back(iter->first);
		}
		else if (temp.size() < k) {
			if (iter->second < cont[temp.front()]) {
				temp.push_front(iter->first);
			}
		}
		
	}*/
	return result;
}

// 31.  Decode String
string decodeString(string s) {
	int i = 0;
	return decodeString(s, i);
}

string decodeString(string s, int& i) {
	string output = "";
	while (i < s.length() && s[i] != ']') {
		if (!isdigit(s[i])) {
			output += s[i];
			i++;
		}
		else {
			int n = 0;
			while (i < s.length() && isdigit(s[i])) {
				n = 10 * n + s[i++] - '0';		// char to int
			}				
			i++;					// ����'['
			string rep = decodeString(s, i);
			i++;					// ����']'
			// �����ظ��ַ���
			while (n > 0) {
				output += rep;
				n--;
			}
		}
	}
	return output;
}

// 32. Queue Reconstruction by Height
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
	// sort �߼�ʹ�÷���
	sort(people.begin(), people.end(), [](vector<int> v1, vector<int> v2) {
		return v1[0] > v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]);
	});
	vector<vector<int>> result;
	/*
		v.insert(v.begin(),8);//����ǰ�������Ԫ��,��ʱvΪ8 2 7 9 5
		v.insert(v.begin()+3,1);//�ڵ��������±�Ϊ3��Ԫ��ǰ������Ԫ��,��ʱvΪ8 2 7 1 9 5
		v.insert(v.end(),3);//������ĩβ׷����Ԫ��,��ʱvΪ8 2 7 1 9 5 3
		v.insert(v.end(),3,0);//��β������3��0,��ʱvΪ8 2 7 1 9 5 3 0 0 0
	*/
	for (auto i : people) {
		result.insert(result.begin() + i[1], i);
	}
	return result;
}

// 33. Partition Equal Subset Sum
bool canPartition(vector<int>& nums) {
	int sum = std::accumulate(nums.begin(), nums.end(), 0);
	if (sum & 1)	return false;
	int target = sum >> 1;				// λ������2�˳���<<kλ==��2��k�η���>>kλ==��2��k�η�
	vector<bool> dp(target + 1, false);
	dp[0] = true;
	for (int num : nums) {
		for (int j = target; j >= num; j--) {
			dp[j] = dp[j] || dp[j - num];
		}
	}
	return dp[target];
}

// 34. Path Sum III
// You are given a binary tree in which each node contains an integer value.
// Find the number of paths that sum to a given value.
int pathSum(TreeNode* root, int sum) {
	unordered_map<int, int> preSum;
	preSum[0] = 1;
	int count = 0;
	pathCounter(root, 0, sum, preSum, count);
	return count;
}

void pathCounter(TreeNode* root, int curSum, int target, unordered_map<int, int>& preSum, int& count) {
	if (root == nullptr)	return;
	curSum += root->val;

	// Ŀ��ʹﵽ||{�ο� 37.Subarray Sum Equals K}
	if (preSum.find(curSum - target) != preSum.end()) {
		count += preSum[curSum - target];
	}

	if (preSum.find(curSum) != preSum.end()) {
		preSum[curSum]++;
	}
	else {
		preSum[curSum] = 1;
	}

	pathCounter(root->left, curSum, target, preSum, count);
	pathCounter(root->right, curSum, target, preSum, count);

	// ��ԭpreSum��¼���ָ���������
	preSum[curSum]--;
}

// 35. Find All Anagrams in a String
// Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
vector<int> findAnagrams(string s, string p) {
	vector<int> result;
	if (s.length() < p.length())	return result;
	// ����p���ַ���
	unordered_map<char, int> pMap;
	for (char c : p) {
		if (pMap.find(c) != pMap.end()) {
			pMap[c]++;
		}
		else {
			pMap[c] = 1;
		}
	}
	int count = pMap.size();					// ��¼target��map��С
	
	int begin = 0, end = 0;

	while (end < s.length()) {
		char c = s[end];
		if (pMap.find(c) != pMap.end()) {
			pMap[c]--;							// ������ʱΪ��ֵ ��beginɨ�赽������
			if (pMap[c] == 0)	count--;		// ĳ���ַ�������
		}
		end++;

		// �ҵ�һ�����
		while (count == 0) {
			char beginC = s[begin];
			if (pMap.find(beginC) != pMap.end()) {
				pMap[beginC]++;
				if(pMap[beginC] == 1)	count++;// ������map���ڳ�mapʱcount++
			}
			if (end - begin == p.length()) {
				result.push_back(begin);
			}
			begin++;	
		}
	}
	return result;
}

// 36. Target Sum
// sum(P) - sum(N) = target => 2 * sum(P) = target + sum(nums)
int findTargetSumWays(vector<int>& nums, int S) {
	int sum = std::accumulate(nums.begin(), nums.end(), 0);		// accumulate �÷���http://www.cplusplus.com/reference/numeric/accumulate/
	return (sum < S || (sum + S) & 1) ? 0 : subsetSum(nums, (sum + S) >> 1);
	// λ�����ж���ż����ĩλΪ1=>������0=>ż��
	// λ������2�˳���<<kλ==��2��k�η���>>kλ==��2��k�η�
}
// ���Ӽ���Ϊtarget��
int subsetSum(vector<int>& nums, int target) {
	vector<int> dp(target + 1, 0);
	dp[0] = 1;
	for (int n : nums) {
		for (int j = target; j >= n; j--) {
			dp[j] += dp[j - n];
		}
	}
	return dp[target];
}

// 37. Subarray Sum Equals K �������к�Ϊk��
// Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.
// sum[i] - sum[j] = k => the sum of elements lying between indices i and j is k.
int subarraySum(vector<int>& nums, int k) {
	unordered_map<int, int> sumMap;
	sumMap[0] = 1;
	int sum = 0, result = 0;
	for (int n : nums) {
		sum += n;
		// update sumMap
		if (sumMap.count(sum)) {
			sumMap[sum]++;
		}
		else {
			sumMap[sum] = 1;
		}
		// sum[i] - sum[j] = k
		if (sumMap.find(sum - k) != sumMap.end()) {
			result += sumMap[sum - k];
		}		
	}
	return result;
}

// 38. Shortest Unsorted Continuous Subarray
int findUnsortedSubarray(vector<int>& nums) {
	int n = nums.size(), tail = -2, front = -1;			// ��֤tail - front == -1����
	int max = nums[0], min = nums[n - 1];
	for (int i = 1; i < n; i++) {
		max = std::max(max, nums[i]);					// ץͷ�ϵ�
		if (nums[i] < max)	tail = i;					// ��β��

		min = std::min(min, nums[n - 1 - i]);			// ץβ�ϵ�
		if (nums[n - 1 - i] > min) front = n - 1 - i;	// ��ͷ
	}
	return tail - front + 1;							// β-ͷ+1
}

// 39. Task Scheduler
int leastInterval(vector<char>& tasks, int n) {
	// ��ĸ�������26�� �ʿ�����vector c[26] ����map
	vector<int> count(26, 0);
	for (char c : tasks) {
		count[c - 'A']++;
	}
	
	std::sort(count.begin(), count.end(), greater<int>());		// ��������
	int maxNum = 0;
	while(maxNum < 26 && count[maxNum] == count[0])	maxNum++;	// ��ִ�������������ж��ٸ�
	// "AB**AB**AB" �ĳ���
	int frame = (count[0] - 1) * (n + 1) + maxNum;
	return std::max((int) tasks.size(), frame);					// .size()����ֵ����Ϊsize_t
}

// 40. Palindromic Substrings
// �һ����Ӵ������Ŀ�����
int countSubstrings(string s) {
	int result = 0;
	if (s.empty()) return result;
	for (int i = 0; i < s.length(); i++) {
		int left = i, right = i;
		for (; left >= 0 && right < s.length() && s[left] == s[right]; left--, right++) {
			result++;
		}
		for (left = i, right = i + 1; left >= 0 && right < s.length() && s[left] == s[right]; left--, right++) {
			result++;
		}
	}
	return result;
}

// 41. Daily Temperatures
vector<int> dailyTemperatures(vector<int>& T) {
	int n = T.size();
	vector<int> result(n, 0);
	for (int i = n - 2; i >= 0; i--) {
		if (T[i] < T[i + 1]) {
			result[i] = 1;
		}
		else {
			int j = i + 1 + result[i + 1];
			bool flag = true;					// ���ҵ���ů��
			while (j < n && T[i] >= T[j]) {
				if (result[j] != 0) {
					j += result[j];
				}
				else {
					flag = false;				// �����ҵ���ů��
					break;
				}				
			}
			if (flag) {
				result[i] = j - i;
			}
		}
	}
	return  result;
}

// 42. Partition Labels
vector<int> partitionLabels(string S) {
	std::vector<int> result;
	if (S.empty())	return result;
	std::unordered_map<char, int> charMap;
	std::vector<pair<int, int>> charVec;
	int index = 0;
	for (int i = 0; i < S.length(); i++) {
		if (charMap.find(S[i]) != charMap.end()) {
			charVec[charMap[S[i]]].second = i;
		}
		else {
			charMap[S[i]] = index++;
			charVec.push_back(make_pair(i, i));
		}
	}

	int front = charVec[0].first, back = charVec[0].second;
	for (int i = 1; i < charVec.size(); i++) {
		if (charVec[i].first < back) {
			if (charVec[i].second > back) {
				back = charVec[i].second;
			}
		}
		else {
			result.push_back(back - front + 1);
			front = charVec[i].first;
			back = charVec[i].second;
		}
	}
	result.push_back(back - front + 1);
	return result;
}

// 43. Valid Sudoku
bool isValidSudoku(vector<vector<char>>& board) {
	for (int i = 0; i < board.size(); i++) {
		for (int j = 0; j < board[0].size(); j++) {
			if (board[i][j] == '.')	continue;
			char c = board[i][j];
			board[i][j] = '.';
			if (!isValidNum(board, i, j, c))	return false;
			board[i][j] = c;
		}
	}
	return true;
}

bool isValidNum(vector<vector<char>>& board, int row, int col, char c) {
	int rowNo = 3 * (row / 3);
	int colNo = 3 * (col / 3);
	for (int i = 0; i < board.size(); i++) {
		if (board[row][i] == c || board[i][col] == c || board[rowNo + i % 3][colNo + i / 3] == c) {
			return false;
		}
	}
	return true;
}

// 44. Multiply Strings
string multiply(string num1, string num2) {
	int m = num1.length(), n = num2.length();
	vector<int> result(m + n, 0);
	// `num1[i] * num2[j]` will be placed at indices `[i + j`, `i + j + 1]` 
	for (int j = m - 1; j >= 0; j--) {
		for (int i = n - 1; i >= 0; i--) {
			int mul = (num1[j] - '0') * (num2[i] - '0');
			result[i + j + 1] += mul % 10;
			result[i + j] += mul / 10;			
		}
	}
	// ��λ
	for (int i = m + n - 1; i >= 0; i--) {
		if (result[i] >= 10) {
			result[i - 1] += result[i] / 10;
			result[i] = result[i] % 10;
		}
	}
	// ��int����ת��Ϊ�ַ��� ͬʱȥ��0
	string resultString = "";
	for (int i = 0; i < result.size(); i++) {
		if (result[i] == 0 && resultString.empty())	continue;
		resultString += result[i] + '0';
	}
	return resultString.empty() ? "0" : resultString;
}

// 45. Wildcard Matching
bool isMatchWildcard(string s, string p) {
	int star = -1;		// '*' ���ֵ�λ��
	int range = -2;		// '*' ������ķ�Χ�Ľ�β����������

	int j = 0;			// p��index
	for (int i = 0; i < s.length(); ) {
		if (p[j] == '?' || s[i] == p[j]) {
			i++;
			j++;
			continue;	// �˳���ǰѭ��
		}

		if (p[j] == '*') {
			star = j++;	// ��¼'*'λ�� ͬʱj����
			range = i;	// ���·�Χβ
			continue;	// �˳���ǰѭ��
		}

		if (star != -1) {
			i = ++range;	// ��Χ����׼����Χ��β��char
			j = star + 1;	// j���û�star����һλ
			continue;
		}

		return false;
	}

	while (j < p.length() && p[j] == '*')	j++;	// ���ʣ���ַ�
	return j == p.length();
}

// 46. Regular Expression Matching
bool isMatchRegular(string s, string p) {
	vector<bool> temp(p.length() + 1, false);
	vector<vector<bool>> dp(s.length() + 1, temp);

	dp[0][0] = true;
	// ��֤ͷ����*
	for (int j = 0; j < p.length(); j++) {
		if (p[j] == '*' && dp[0][j - 1]) {
			dp[0][j + 1] = true;
		}
	}
	// 
	for (int i = 0; i < s.length(); i++) {
		for (int j = 0; j < p.length(); j++) {
			if (s[i] == p[j] || p[j] == '.') {
				dp[i + 1][j + 1] = dp[i][j];
			}
			else if(p[j] == '*'){
				if (s[i] != p[j - 1] && p[j - 1] != '.') {		// ע��j-1λ���ϲ���Ϊ'.'
					dp[i + 1][j + 1] = dp[i + 1][j - 1];
				}
				else {
					// multiple a || single a || empty
					dp[i + 1][j + 1] = (dp[i][j + 1] || dp[i + 1][j] || dp[i + 1][j - 1]);
				}
			}
		}
	}
	return dp[s.length()][p.length()];
}

// 47. Permutations II
vector<vector<int>> permuteUnique_Backtracking(vector<int>& nums) {
	vector<vector<int>> result;
	if (nums.empty())	return result;
	
	std::sort(nums.begin(), nums.end());
	vector<bool> isUsed(nums.size(), false);
	vector<int> item;
	DFS_permute(nums, isUsed, item, result);
	return result;
}

void DFS_permute(vector<int>& nums, vector<bool>& isUsed, vector<int> item, vector<vector<int>>& result) {
	if (item.size() == nums.size()) {
		result.push_back(item);
		return;
	}
	for (int i = 0; i < nums.size(); i++) {
		if (isUsed[i])	continue;	
		// when a number has the same value with its previous, we can use this number only if his previous is used
		if (i > 0 && nums[i] == nums[i - 1] && !isUsed[i - 1])	continue;
		isUsed[i] = true;
		item.push_back(nums[i]);
		DFS_permute(nums, isUsed, item, result);
		item.pop_back();	
		isUsed[i] = false;
	}
}

// ������
vector<vector<int>> permuteUnique_Swap(vector<int>& nums) {
	vector<vector<int>> result;
	if (nums.empty())	return result;
	Swap_permute(0, nums, result);
	return result;
}

void Swap_permute(int begin, vector<int> nums, vector<vector<int>>& result) {
	
	if (begin == nums.size()) {
		result.push_back(nums);
		return;
	}
	for (int i = begin; i < nums.size(); i++) {
		if (nums[i] == nums[begin] && i != begin)	continue;
		swap(nums[begin], nums[i]);
		Swap_permute(begin + 1, nums, result);
	}
}

// 48. Pow(x,n)
// O(nlogN) ���η�
double myPow(double x, int n) {
	if (n == 0)	return 1;
	if (n < 0) {
		x = 1 / x;
		if (n == INT_MIN) {
			n = INT_MAX;			// INT_MIN û�ж�Ӧ�ķǸ���
			return x * myPow(x, n);
		}
		n = -n;		
	}
	return (n & 1) ? (x * myPow(x * x, n / 2)) : myPow(x * x, n / 2);
}

// 49. N�ʺ�
vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> result;
	vector<string> item(n, string(n, '.'));
	nQueensSolver(0, n, item, result);
	return result;
}

void nQueensSolver(int row, int n, vector<string>& item, vector<vector<string>>& result) {
	if (row == n) {
		result.push_back(item);
		return;
	}
	for (int col = 0; col < n; col++) {
		item[row][col] = 'Q';
		if (isValidQueen(item, row, col, n)) {
			nQueensSolver(row + 1, n, item, result);
		}
		item[row][col] = '.';
	}
}

bool isValidQueen(vector<string> item, int row, int col, int n) {
	for (int i = 0; i < row; i++) {
		if (item[i][col] == 'Q')	return false;
	}
	for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
		if (item[i][j] == 'Q')	return false;
	}
	for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
		if (item[i][j] == 'Q')	return false;
	}
	return true;
}

// 50. Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	vector<int> result;
	int rowBegin = 0, rowEnd = matrix.size() - 1;
	int colBegin = 0, colEnd = matrix[0].size() - 1;

	while (rowBegin <= rowEnd && colBegin <= colEnd) {
		// ����
		for (int i = colBegin; i <= colEnd; i++) {
			result.push_back(matrix[rowBegin][i]);
		}
		rowBegin++;

		// ����
		for (int i = rowBegin; i <= rowEnd; i++) {
			result.push_back(matrix[i][colEnd]);
		}
		colEnd--;

		// ����(���ڵ�ǰ��Υ��ɨ���ʱִ�У�rowBegin <= rowEnd)
		if (rowBegin <= rowEnd) {
			for (int i = colEnd; i >= colBegin; i--) {
				result.push_back(matrix[rowEnd][i]);
			}
			rowEnd--;
		}
		
		// ����(���ڵ�ǰ��Υ��ɨ���ʱִ�У�colBegin <= colEnd)
		if (colBegin <= colEnd) {
			for (int i = rowEnd; i <= rowBegin; i--) {
				result.push_back(matrix[i][colBegin]);
			}
			colBegin++;
		}		
	}
	return result;
}

// 51. Add Strings
string addStrings(string num1, string num2) {
	if (num1.empty() && num2.empty()) {
		return "";
	}
	int m = num1.length(), n = num2.length();
	int resultLength;
	if (m < n) {
		swap(num1, num2);
		swap(m, n);
	}
	resultLength = m + 1;
	
	vector<int> addOn(resultLength, 0);
	for (int i = 0; i < n; i++) {
		addOn[resultLength - 1 - i] = num1[m - 1 - i] + num2[n - 1 - i] - 2 * '0';
	}
	for (int i = 0; i < m - n; i++) {
		addOn[i + 1] = num1[i] - '0';
	}

	for (int i = resultLength - 1; i >= 0; i--) {
		if (addOn[i] >= 10) {
			addOn[i] = addOn[i] % 10;
			addOn[i - 1]++;
		}
	}

	string result = "";
	int begin = addOn[0] == 0 ? 1 : 0;
	while (begin < resultLength)
	{
		result += '0' + addOn[begin];
		begin++;
	}
	
	return result;
}

// 52. LRU Cache
int capacity;
unordered_map<int, DbListNode*>  keyMap;
DbListNode Head(make_pair(0, 0)), Tail(make_pair(0, 0), &Head);
void LRUCache(int val) {
	capacity = val;		
	Head.next = &Tail;
}

int get(int key) {
	if (keyMap.find(key) != keyMap.end()) {
		DbListNode* tmp = keyMap[key];
		if (tmp == Head.next)	return tmp->val.second;
		tmp->front->next = tmp->next;
		tmp->next->front = tmp->front;

		tmp->next = Head.next;
		tmp->front = &Head;
		Head.next->front = tmp;
		Head.next = tmp;
		return tmp->val.second;
	}
	else {
		return -1;
	}
}

void put(int key, int value) {
	// ��ֵ���Ѵ��� �����
	if (keyMap.find(key) != keyMap.end()) {
		get(key);
		keyMap[key]->val.second = value;
	}
	else {
		if (keyMap.size() >= capacity) {
			// ɾ��β�ڵ� ���key
			DbListNode* endNode = Tail.front;
			endNode->front->next = &Tail;
			Tail.front = endNode->front;
			// ���±�
			keyMap.erase(endNode->val.first);
			delete(endNode);
		}
		// �����½ڵ㵽Head��
		DbListNode* node = new DbListNode(make_pair(key, value));
		keyMap[key] = node;
		node->next = Head.next;
		Head.next->front = node;
		Head.next = node;
		node->front = &Head;
	}	
}