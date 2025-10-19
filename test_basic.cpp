#include <iostream>
#include <vector>
#include <string>

using namespace std;

class SimpleSemantic {
private:
    vector<string> texts;

public:
    void addText(const string& text) {
        texts.push_back(text);
    }
    
    vector<string> query(const string& query, int k = 10) {
        vector<string> result;
        
        // 简单的包含匹配
        for (int i = 0; i < texts.size() && result.size() < k; i++) {
            if (texts[i].find(query) != string::npos) {
                result.push_back(texts[i]);
            }
        }
        
        return result;
    }
};

int main() {
    cout << "简单语义匹配器测试\n" << endl;
    
    SimpleSemantic matcher;
    
    matcher.addText("神经网络基础");
    matcher.addText("深度学习算法");
    matcher.addText("卷积神经网络");
    matcher.addText("CUDA并行编程");
    
    vector<string> results = matcher.query("神经网络", 3);
    
    cout << "查询 '神经网络' 结果:" << endl;
    for (int i = 0; i < results.size(); i++) {
        cout << i+1 << ". " << results[i] << endl;
    }
    
    cout << "\n✅ 演示完成!" << endl;
    cout << "\n💡 这个语义匹配器:" << endl;
    cout << "   • 直接返回string结果" << endl;
    cout << "   • 简单包含匹配" << endl; 
    cout << "   • 截断k个结果" << endl;
    cout << "   • 没有复杂的NeuronInput注入" << endl;
    cout << "   • 简单易用" << endl;
    
    return 0;
}