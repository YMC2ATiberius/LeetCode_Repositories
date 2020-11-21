#include <August.h>

using namespace std;

//// 1.Detect Capital
bool detectCapitalUse(std::string word) {
	if (word.empty()) {
		return false;
	}
	bool isFirstCaptical = false;
	bool isSecondCaptital = false;

	if (word[0] <= 90) {
		isFirstCaptical = true;
	}
	for (int i = 1; i < word.size(); i++) {
		if (!isFirstCaptical) {
			if (word[i] <= 90) {
				return false;
			}
		}
		else {
			isSecondCaptital = (word[1] <= 90);
			if (!isSecondCaptital) {
				if (word[i] <= 90) {
					return false;
				}
			}
			else {
				if (word[i] >= 97) {
					return false;
				}
			}
		}
	}
	return true;
}

// 2.Design HashSet
//class MyHashSet {
//public:
//	
//	void add(int key) {
//		hashMap[key] = true;
//	}
//
//	void remove(int key) {
//		if (hashMap[key]) {
//			hashMap[key] = false;
//		}
//	}
//
//	/** Returns true if this set contains the specified element */
//	bool contains(int key) {
//		return hashMap[key];
//	}
//private:
//	bool hashMap[1000001] = { false };
//};



// 3. Valid Palindrome
bool isPalindrome(string s) {
	if (s.length() < 2)
		return true;

	int i = 0, j = s.length() - 1;
	while (i < j) {
		if (isalnum(s[i]) && isalnum(s[j])) {
			if (tolower(s[i]) != tolower(s[j])) {
				return false;
			}
			i++;
			j--;
		}
		else if(!isalnum(s[i]))
		{
			i++;
		}
		else
		{
			j--	;
		}
	}
	return true;
}

// 4.Power of Four
bool isPowerOfFour(int num) {	
	if (num <= 0) {
		return false;
	}
	else if(num == 1)
	{
		return true;
	}
	else {
		while (num > 4) {
			if (num % 4 != 0) {
				return false;
			}
			num = num / 4;
		}
	}	
	return num % 4 == 0;
}

// 5. [15]3Sum
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> result;
	if (nums.size() < 3) {
		return result;
	}
	int maxIndex = nums.size() - 1;
	vector<int> candidate;
	std::sort(nums.rbegin(), nums.rend());
	for (int i = 0; i < maxIndex - 1; i++) {
		int j = i + 1, k = maxIndex;
		while (j < k) {
			int sum = nums[i] + nums[j];
			if (sum + nums[k] < 0) {
				k--;
			}
			else if (sum + nums[k] > 0) {
				j++;
			}
			else {
				candidate.push_back(nums[i]);
				candidate.push_back(nums[j]);
				candidate.push_back(nums[k]);
				result.push_back(candidate);
				// move i and j
				// and remove duplicate of 2nd and 3rd
				while (j < k && candidate[1] == nums[j]) j++;
				while (j < k && candidate[2] == nums[k]) k--;
				candidate.clear();
			}
		}
		// remove duplicate of 1st
		while (i < maxIndex - 1 && nums[i + 1] == nums[i]) i++;
	}

	return result;
}

// 5.1 3Sum Closest
int threeSumClosest(vector<int>& nums, int target) {
	int dis = INT_MAX, sum = 0, result = 0;
	std::sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size() - 2; i++) {
		int j = i + 1, k = nums.size() - 1;
		while (j < k) {
			sum = nums[i] + nums[j] + nums[k];
			int new_dis = abs(sum - target);
			if (new_dis < dis) {
				dis = new_dis;
				result = sum;
			}
			else if (new_dis == 0) {
				return sum;
			}
			// Go for new i-j-k set
			if (sum < target) {
				j++;
			}
			else {
				k--;
			}
		}

	}
	return result;
}

// 6. Letter Combinations of a Phone Number
vector<string> letterCombinations(string digits) {
	vector<string> result;
	if (!digits.size()) {
		return result;
	}	
	string word;
	unordered_map<char, string> phone;
	buildPhone(phone);
	int level = 0;
	creatWord(phone, result, word, digits, level);	
	return result;
}

void creatWord(unordered_map<char, string>& phone, vector<string> &result, string &word, string &digits, int &level) {
	if (level >= digits.size()) {
		result.push_back(word);
		return;
	}
	// str: 每次要搜索的字符
	string str = phone[digits[level]];
	level++;
	for (int i = 0; i < str.size(); i++) {
		word.push_back(str[i]);		
		creatWord(phone, result, word, digits, level);
		word.pop_back();		
	}
	level--;
}

void buildPhone(unordered_map<char,string>& phone) {
	int dic, leng;
	string str;
	char i;
	for (i = '2'; i <= '7'; i++) {		
		dic = i - '2';
		leng = i == '7' ? 4 : 3;
		for (int j = 0; j < leng; j++) {
			str.push_back('a' + 3 * dic + j);
		}
		phone[i] = str;
		str.clear();
	}
	for (i = '8'; i <= '9'; i++) {
		dic = i - '8';
		leng = i == '9' ? 4 : 3;
		for (int j = 0; j < leng; j++) {
			str.push_back('t' + 3 * dic + j);
		}
		phone[i] = str;
		str.clear();
	}
}

// 7. Valid Parentheses
bool isValid(string s) {
	if (s.size() == 0) {
		return true;
	}
	if (s.size() == 0 || s.size() % 2) {
		return false;
	}
	bool doInput = false;
	if (s[0] == ')' || s[0] == '}' || s[0] == ']') {
		return false;
	}
	std::vector<char> parentheses;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '(' || s[i] == '{' || s[i] == '[') {
			parentheses.push_back(s[i]);
		}
		else
		{
			char temp = parentheses.back();			
			switch (s[i])
			{
			case ')':
				if (temp != '(')
					return false;
				break;
			case ']':
				if (temp != '[')
					return false;
				break;
			case '}':
				if (temp != '{')
					return false;
				break;
			default:
				break;
			}
			parentheses.pop_back();
		}
	}
	return parentheses.empty();
}

// 8. 4Sum
vector<vector<int>> fourSum(vector<int>& nums, int target) {
	vector<vector<int>> result;
	std::sort(nums.begin(), nums.end());
	int maxIndex = nums.size() - 1;
	for (int i = 0; i < maxIndex - 2; i++) {
		if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
		if (nums[i] + nums[maxIndex] + nums[maxIndex - 1] + nums[maxIndex - 2] < target) continue;
		for (int j = i + 1; j < maxIndex - 1; j++) {
			if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) break;
			if (nums[i] + nums[j] + nums[maxIndex] + nums[maxIndex - 1] < target) continue;
			int left = j + 1, right = maxIndex;
			while (left < right) {
				int sum = nums[i] + nums[j] + nums[left] + nums[right];
				if (sum < target) {
					left++;
				}
				else if (sum > target) {
					right--;
				}
				else {
					vector<int> candidate(4);
					candidate[0] = nums[i];
					candidate[1] = nums[j];
					candidate[2] = nums[left];
					candidate[3] = nums[right];
					result.push_back(candidate);
					do left++; while (left < right && candidate[2] == nums[left]);
					do right--; while (left < right && candidate[3] == nums[right]);
				}
			}
			while (j < maxIndex - 1 && nums[j] == nums[j + 1]) j++;
		}
		while (i < maxIndex - 2 && nums[i] == nums[i + 1]) i++;
	}
	return result;
}

// 9. Remove Nth Node From End of List
ListNode* removeNthFromEnd(ListNode* head, int n) {	
	if (!head) {
		return nullptr;
	}

	ListNode* pre = nullptr;
	ListNode* front = head;
	ListNode* back = head;

	/*直接选出删除区间*/
	while (n-- && back) {
		back = back->next;
	}

	while (back) {
		pre = front;
		front = front->next;
		back = back->next;
	}
	// Delete first element
	if (front == head) {
		head = front->next;
		delete(front);
		return head;
	}
	pre->next = front->next;
	delete(front);
	return head;

	// 慢方法
	/*vector<ListNode*> nodeVec;
	ListNode* node = head;
	while (node) {
		nodeVec.push_back(node);
		node = node->next;
	}
	ListNode* temp;
	int length = nodeVec.size();
	if (n == length) {
		temp = head;
		head = head->next;
	}
	else {
		temp = nodeVec[length - n - 1];
		ListNode* del = temp->next;
		temp->next = del->next;
	}
	return head;*/
}

// 10. 24. Swap Nodes in Pairs
ListNode* swapPairs(ListNode* head) {
	ListNode* front = new ListNode(1);
	ListNode* pre = front;
	pre->next = head;
	
	while (pre->next && pre->next->next) {
		ListNode* first = pre->next;
		ListNode* second = first->next;
		pre->next = second;
		first->next = second->next;
		second->next = first;
		pre = first;
	}
	return front->next;
}

// 11. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, int k) {
	ListNode front(0);
	front.next = head;
	ListNode* pre_head = head;
	bool isFirst = true;
	while (head) {
		int count = k;
		ListNode* tail = head;
		while (count > 1) {			
			tail = tail->next;
			count--;
			if (!tail) {
				return front.next;
			}
		}
		ListNode* next_head = tail->next;
		reverseNodes(head, tail);
		if (isFirst) {
			front.next = head;
			isFirst = false;
		}
		pre_head->next = head;
		pre_head = tail;
		tail->next = next_head;
		head = next_head;
	}
	return front.next;
}

void reverseNodes(ListNode* &head, ListNode* &tail) {
	ListNode* new_head = nullptr;
	tail->next = nullptr;
	tail = head;
	while (head) {
		ListNode* temp = head->next;
		head->next = new_head;
		new_head = head;
		head = temp;
	}
	head = new_head;
}

// 12. Longest Common Prefix
string longestCommonPrefix(vector<string>& strs) {
	
	if (!strs.size()) {
		return "";
	}
	string result = strs[0];
	for (int i = 1; i < strs.size(); i++) {		
		string word = strs[i];
		string candidate = "";
		for (int j = 0; j < result.size(); j++) {
			if (word[j] == result[j]) {
				candidate.append(word.substr(j, 1));
			}
			else {
				break;
			}
		}		
		result = candidate;
	}
	return result;
};

// 13. Longest Valid Parentheses
int longestValidParentheses(string s) {
	// push -1 记录第一个（的前一位置
	int result = 0;	
	vector<int> pare;
	pare.push_back(-1);
	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '(') {
			pare.push_back(i);
		}
		else {			
			pare.pop_back();
			if (!pare.empty()) {
				int length = i - pare.back();
				result = max(length, result);
			}
			else {
				pare.push_back(i);
			}
		}
	}
	return result;
}

// 14. Next Permutation
void nextPermutation(vector<int>& nums) {
	bool swap = false;
	for (int i = nums.size() - 1; i >= 0; i--) {		
		if (i > 0 && nums[i - 1] < nums[i]) {
			replace(nums, i, nums[i - 1]);
			std::sort(nums.begin() + i, nums.end());
			swap = true;
			break;
		}
	}
	if (!swap) {
		std::sort(nums.begin(), nums.end());
	}
}

void replace(vector<int>& nums, int begin, int target) {	
	for (int i = nums.size() - 1; i >= begin; i--) {
		if (nums[i] > target) {
			int temp = nums[i];
			nums[i] = target;
			nums[begin - 1] = temp;
			break;
		}
	}
}

// 15. Combination Sum
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
	std::sort(candidates.rbegin(), candidates.rend());
	vector<int> item;
	vector<vector<int>> result;

	for (int i = 0; i < candidates.size(); i++) {		
		int j = searchItem(candidates, target, i-1, item, result);
		i = j;
	}
	return result;
}

int searchItem(vector<int>& candidates, int &target, int i, vector<int>& item, vector<vector<int>>& result) {
	if (target == 0) {
		result.push_back(item);
		return i;
	}
	i++;
	while (i < candidates.size()) {		
		if (candidates[i] < target) {
			int repTime = target / candidates[i];
			for (int k = 0; k < repTime; k++) {
				item.push_back(candidates[i]);
				target = target - candidates[i];
				searchItem(candidates, target, i, item, result);
			}
			while (repTime > 0) {
				item.pop_back();
				target = target + candidates[i];
				repTime--;
			}
		}
		else if (candidates[i] == target) {
			item.push_back(candidates[i]);
			result.push_back(item);
			item.pop_back();
		}
		i++;
	}
	return i;
}

// 16. Trapping Rain Water
int trap(vector<int>& height) {
	int length = height.size();
	if (length < 3) {
		return 0;
	}

	int level = min(height[0], height[length - 1]);
	int i = 0, j = length - 1;
	int result = 0;
	
	while (i < j) {
		if (height[i] <= height[j]) {
			if (height[i] <= level) {
				result += level - height[i];
			}
			else {
				level = height[i];
			}
			i++;
		}
		else {
			if (height[j] <= level) {
				result += level - height[j];
			}
			else {
				level = height[j];
			}
			j--;
		}
	}
	return result;
}

// 17. First Missing Positive
int firstMissingPositive(vector<int>& nums) {
	set<int> bag;
	int floor = 1;
	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] > 0) {			
			bag.insert(nums[i]);
			if (nums[i] == floor || nums[i] - floor == 1) {
				while (bag.find(floor) != bag.end()) {
					floor++;
				}
			}			
		}
	}
	return  floor;
}

// 18. Permutations
vector<vector<int>> permute(vector<int>& nums) {
	vector <vector<int>> result;

	generatePermutation(result, nums, 0);
	return result;
}
// E:\@ LeetCode课件\NewPermutation.gif
void generatePermutation(vector<vector<int>> &result, vector<int> &nums, int begin) {
	if (begin == nums.size()) {
		result.push_back(nums);
		return;
	}
	for (int i = begin; i < nums.size(); i++) {
		swap(nums[begin], nums[i]);
		generatePermutation(result, nums, begin + 1);
		swap(nums[begin], nums[i]);
	}
}

// 19. Rotate Image
void rotate(vector<vector<int>>& matrix) {
	int row = matrix.size();
	int col = matrix[0].size();

	reverse(matrix.begin(), matrix.end());
	for (int i = 0; i < row; i++) {
		for (int j = 0; j <= i; j++) {
			swap(matrix[i][j], matrix[j][i]);			
		}
	}	
}

// 20. Merge Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {
	vector<vector<int>> result;
	if (intervals.empty()) {
		return result;
	}

	std::sort(intervals.begin(), intervals.end());	
	vector<int> item = intervals[0];

	for (int i = 1; i < intervals.size(); i++) {
		if (intervals[i][0] > item[1]) {
			result.push_back(item);
			item = intervals[i];
		}
		else if (intervals[i][1] > item[1]) {
			item[1] = intervals[i][1];
		}
	}
	result.push_back(item);
	return result;
}

// 21. Unique Paths
int uniquePaths(int m, int n) {
	int N = m + n - 2;
	int K = min(m - 1, N - (m - 1));
	long result = 1.0;

	for (int i = 1; i <= K; i++) {
		result = result * (N - K + i) / i;
	}
	return (int) result;
}
/* 
	DP：
	Since the robot can only move right and down, when it arrives at a point, it either arrives from left or above. 
	If we use dp[i][j] for the number of unique paths to arrive at the point (i, j), 
	then the state equation is dp[i][j] = dp[i][j - 1] + dp[i - 1][j]. 
	Moreover, we have the base cases dp[0][j] = dp[i][0] = 1 for all valid i and j.

	However, you may have noticed that each time when we update dp[i][j], 
	we only need dp[i - 1][j] (at the previous row) and dp[i][j - 1] (at the current row). 
	So we can reduce the memory usage to just two rows (O(n)).
	还要学学！！
*/
int uniquePaths_DP(int m, int n) {
	vector<int> pre(n, 1), cur(n, 1);
	for (int i = 1; i < m; i++) {
		for (int j = 1; j < n; j++) {
			cur[j] = pre[j] + cur[j - 1];
		}
		swap(pre, cur);
	}
	return pre[n - 1];
}

// 22. Edit Distance
int minDistance(string word1, string word2) {
	int m = word1.length(), n = word2.length();
	vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 1; i <= m; i++) {
		dp[i][0] = i;
	}
	for (int i = 1; i <= n; i++) {
		dp[0][i] = i;
	}
	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= n; j++) {
			if (word1[i - 1] == word2[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
			}
		}
	}
	return dp[m][n];
}

int minDistance_twoVec(string word1, string word2) {
	int m = word1.length(), n = word2.length();
	vector<int> pre(n + 1,0), cur(n + 1,0);
	for (int i = 1; i <= n; i++) {
		pre[i] = i;
	}
	for (int i = 1; i <= m; i++) {
		cur[0] = i;
		for (int j = 1; j <= n; j++) {
			if (word1[i - 1] == word2[j - 1]) {
				cur[j] = pre[j-1];
			}
			else {
				cur[j] = min(pre[j - 1], min(pre[j], cur[j - 1])) + 1;
			}
		}
		fill(pre.begin(), pre.end(), 0);
		swap(pre, cur);
	}
	return pre[n];
}

// 23. Sort Colors
void sortColors(vector<int>& nums) {
	// 移位法,注意顺序2>1>0
	int length = nums.size();
	int n0 = -1, n1 = -1, n2 = -1;
	for (int i = 0; i < length; i++) {
		if (nums[i] == 0) {
			nums[++n2] = 2;
			nums[++n1] = 1;
			nums[++n0] = 0;

		}
		else if (nums[i] == 1) {
			nums[++n2] = 2;
			nums[++n1] = 1;
		}
		else {
			nums[++n2] = 2;
		}
	}

	// 快速排序法
	// quick_sort(nums, 0, nums.size() - 1);
}

// 24 Word Search
bool exist(vector<vector<char>>& board, string word) {
	int m = board.size(), n = board[0].size();
	vector<vector<int>> mark(m, vector<int>(n, 0));

	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			if (board[row][col] == word[0]) {
				mark[row][col] = 1;
				int w = 1;				
				if (w == word.size() || searchWord(board, mark, word, row, col, w)) {
					return true;
				}
				mark[row][col] = 0;
			}
		}
	}
	return false;
}

bool searchWord(vector<vector<char>>& board, vector<vector<int>> &mark, string word, int row, int col, int w) {	
	const int dm[] = { 0, 0, 1, -1 };
	const int dn[] = { 1, -1, 0, 0 };
	for (int i = 0; i < 4; i++) {
		int new_row = row + dm[i];
		int new_col = col + dn[i];
		if (new_row < board.size() && new_col < board[0].size() && new_row >= 0 && new_col >= 0) {
			if (board[new_row][new_col] == word[w] && mark[new_row][new_col] == 0) {
				mark[new_row][new_col] = 1;
				if (w + 1 == word.size() || searchWord(board, mark, word, new_row, new_col, w + 1)) {
					return true;
				}
				mark[new_row][new_col] = 0;
			}
		}
	}
	return false;
}

// 25. Largest Rectangle in Histogram
int largestRectangleArea(vector<int>& heights) {
	if (heights.empty()) {
		return 0;
	}	
	return getMaxArea(heights, 0, heights.size() - 1);
}

// 分治法O(NlogN)
int getMaxArea(vector<int>& heights, int left, int right) {
	if (left == right) {
		return heights[left];
	}
	int mid = left + (right - left) / 2;
	int areaLeft = getMaxArea(heights, left, mid);			// 10,1,2
	int areaRight = getMaxArea(heights, mid + 1, right);	// 1,2,10
	int areaMid = getMidArea(heights, left, mid, right);	// 1,2,3
	return max(areaLeft, max(areaMid, areaRight));
}

// 求包含中间的区间值
int getMidArea(vector<int>& heights, int left, int mid, int right) {
	int i = mid;
	int j = mid + 1;
	int minHeight = min(heights[i], heights[j]);
	int area = minHeight * 2;
	while (i >= left && j <= right) {
		minHeight = min(minHeight, min(heights[i], heights[j]));
		area = max(area, minHeight * (j - i + 1));
		if (i == left) {
			j++;
		}
		else if (j == right) {
			i--;
		}
		// 选较高的柱子
		else if (heights[i - 1] >= heights[j + 1]) {
			i--;
		}
		else
		{
			j++;
		}		
	}
	return area;
}

// O(N^2)
int innerCount(vector<int>& heights, int begin) {
	int count = 1;
	for (int i = begin; i < heights.size() - 1; i++) {
		if (heights[begin] <= heights[i + 1]) {
			count++;
		}
		else {
			break;
		}
	}
	for (int i = begin; i > 0; i--) {
		if (heights[i - 1] >= heights[begin]) {
			count++;
		}
		else {
			break;
		}
	}
	return count;
}

// O(N) N^2优化
int largestRectangleArea_N(vector<int>& heights) {
	if (heights.empty()) {
		return 0;
	}

	vector<int> leftLessMin(heights.size(), -1);	
	// left 从左往右
	for (int i = 1; i < heights.size(); i++) {
		int left = i - 1;
		while (left > -1 && heights[left] >= heights[i]) {
			left = leftLessMin[left];
		}
		// 结果赋值
		leftLessMin[i] = left;		
	}
	vector<int> rightLessMin(heights.size(), heights.size());
	// right 从右往左
	for (int i = heights.size() - 2; i > -1; i--) {
		int right = i + 1;
		while (right < heights.size() && heights[right] >= heights[i]) {
			right = rightLessMin[right];
		}
		rightLessMin[i] = right;
	}
	
	int result = 0;
	for (int i = 0; i < heights.size(); i++) {
		result = max((rightLessMin[i] - leftLessMin[i] - 1) * heights[i], result);
	}
	return result;
}

// 快速排序
void quick_sort(vector<int>& nums, int left, int right) {
	if (left < right) {
		swap(nums[left], nums[(left + right) / 2]);
		int i = left, j = right;
		int ref = nums[left];
		while (i < j) {
			while (i < j && nums[j] >= ref) {
				j--;
			}
			if (i < j) {
				nums[i] = nums[j];
				i++;
			}
			while (i < j && nums[i] < ref)
			{
				i++;
			}
			if (i < j) {
				nums[j] = nums[i];
				j--;
			}
		}
		nums[i] = ref;
		quick_sort(nums, left, i - 1);
		quick_sort(nums, i + 1, right);
	}
}

