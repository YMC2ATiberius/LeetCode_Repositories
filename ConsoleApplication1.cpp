// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。

#include <Top100.h>
#include <Cp8Search.h>

using namespace std;

int main()
{
    //vector<vector<char>> test = { {'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'} };
    //vector<vector<char>> test = { {'A'} };
    //vector<int> test = { 2,1,5,6,2,3 };
   //vector<vector<char>> test = { {'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'} };
   /* test.push_back(6);
    test.push_back(2);
    test.push_back(1);
    test.push_back(5);
    test.push_back(4);
    test.push_back(3);
    test.push_back(0);*/
    //test.push_back(7);

    //TreeNode a(10);
    //TreeNode b(5);
    //TreeNode c(15);
    //TreeNode d(6);
    //TreeNode e(20);

    TreeNode a(1);    
    TreeNode b(-2);
    TreeNode c(-3);

    a.left = &b;
    a.right = &c;

    /*a.left = &b;
    a.right = &c;
    c.left = &d;
    c.right = &e;

    vector<int> preoder = { 3,9,20,15,7 };
    vector<int> inorder = { 9,3,15,20,7 };*/
    
    //bool result = isValidBST(&a);
    //TreeNode* result = buildTree(preoder, inorder);

    /*vector<vector<char>> board = { {'o','a','a','n'},{'e','t','a','e'},{'i','h','k','r'},{'i','f','l','v'} };
    vector<string> words = { "oath", "pea", "eat", "rain" };
    vector<string> result = findWords(board, words);*/

    /*vector<vector<char>> matrix = { {'1','0'} };
    int result = maximalSquare(matrix);*/
    /*int a = 0, b = 5;
    int c = a ^ b;*/
    //int result = longestConsecutiveN(nums);

    
    /*vector<vector<char>> board = 
        { {'5', '3', '.', '.', '7', '.', '.', '.', '.'}
        , {'6', '.', '3', '1', '9', '5', '.', '.', '.'}
        , {'.', '9', '8', '.', '.', '.', '.', '6', '.'}
        , {'8', '.', '.', '.', '6', '.', '.', '.', '3'}
        , {'4', '.', '.', '8', '.', '3', '.', '.', '1'}
        , {'7', '.', '.', '.', '2', '.', '.', '.', '6'}
        , {'.', '6', '.', '.', '.', '.', '2', '8', '.'}
        , {'.', '.', '.', '4', '1', '9', '.', '.', '5'}
        , {'.', '.', '.', '.', '8', '.', '.', '7', '9'} };
    bool result = isValidSudoku(board);*/

    vector<int> nums = { 1,2,3 };
    vector<vector<int>> result = permuteUnique_Swap(nums);
    cout << "endl" << endl;


    /*string test = ")()())";
    int result = longestValidParentheses(test);
    cout << "result is:" << result << endl;*/

    /*vector<string> test;
    test.push_back("flower");
    test.push_back("flow");
    test.push_back("flight");
    string result = longestCommonPrefix(test);*/
    

    /*ListNode a(1);
    ListNode b(2);
    ListNode c(3);
    ListNode d(4);
    ListNode e(5);
    a.next = &b;
    b.next = &c;
    c.next = &d;
    d.next = &e;
    ListNode* result = reverseKGroup(&a, 3);*/
   
    //cout << detectCapitalUse("FlaG") << endl;
    //string s = "()[[[]]]";
    //letterCombinations(digits);
    //cout << isValid(s) << endl;


    /*int target = 0;
    vector<int> nums;
    nums.push_back(1);
    nums.push_back(0);
    nums.push_back(-1);
    nums.push_back(0);
    nums.push_back(-2);
    nums.push_back(2);
    cout << "Build is done!" << endl;
    vector<vector<int>> result = fourSum(nums, target);
    cout << "Work is done!" << endl;*/

    /*int s = -64;
    cout << isPowerOfFour(s) << endl;*/

    /*vector<int> test;
    test.push_back(0);
    test.push_back(2);
    test.push_back(1);
    test.push_back(2);
    test.push_back(5);
    test.push_back(4);
    test.push_back(8);
    test.push_back(3);
    test.push_back(7);
    cout << maxArea(test) << endl;*/
   
    /*string beginWord = "hit";
    string endWord = "cog";
    vector<string> word_map;
    word_map.push_back("hot");
    word_map.push_back("dot");
    word_map.push_back("dog");
    word_map.push_back("lot");
    word_map.push_back("log");
    word_map.push_back("cog");
    cout << "Ladder length is:" << ladderLength(beginWord, endWord, word_map)<<endl;*/
    //vector<string> test_result = findRepeatedDnaSequences(s);
    ////int i = 14; //1110  111000 8+16+32
    //std::cout << "process end: " << endl;
    ////std::cout << "test result is: " << findRepeatedDnaSequences(s) << endl;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
