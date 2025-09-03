import streamlit as st
import pandas as pd
import json
from datetime import datetime, date
from collections import defaultdict
import base64
import io
import re
import PyPDF2
import anthropic
import os

# ページ設定
st.set_page_config(
    page_title="月次業務報告レポート生成 (AI Enhanced)",
    page_icon="📊",
    layout="wide"
)

def check_anthropic_setup():
    """Anthropic API設定をチェック"""
    api_key = st.session_state.get('anthropic_api_key') or os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        st.error("Anthropic API KEYが設定されていません")
        with st.expander("API KEY設定方法"):
            st.write("""
            以下のいずれかの方法でAPI KEYを設定してください：
            
            1. **環境変数で設定:**
            ```bash
            export ANTHROPIC_API_KEY="your-api-key-here"
            ```
            
            2. **サイドバーで入力:** (推奨)
            左のサイドバーでAPI KEYを入力してください。
            
            **API KEYの取得:**
            - [Anthropic Console](https://console.anthropic.com/)でアカウント作成
            - API KEYを生成してコピー
            """)
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Anthropic API接続エラー: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """PDFからテキストを抽出"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"--- ページ {page_num + 1} ---\n{page_text}\n\n"
        
        return text
        
    except Exception as e:
        st.error(f"PDF読み取りエラー: {e}")
        return None

def analyze_pdf_with_anthropic(client, pdf_text, filename=""):
    """Anthropic Claude APIでPDFを分析"""
    
    analysis_prompt = f"""
以下は建物管理の点検報告書「{filename}」から抽出されたテキストです：

{pdf_text}

この点検報告書を建物管理の専門家として詳細に分析し、以下のJSON形式で回答してください：

{{
  "inspection_summary": {{
    "inspection_date": "点検実施日（YYYY/MM/DD形式）",
    "contractor": "点検実施業者名",
    "property": "対象物件名", 
    "inspection_type": "点検種別（月次点検、年次点検等）",
    "inspector": "点検者名"
  }},
  "inspection_items": [
    {{
      "item_number": "項目番号", 
      "category": "点検カテゴリ（給排水、電気、空調等）",
      "item_name": "具体的な点検項目名",
      "result": "点検結果の詳細",
      "status": "正常/要注意/異常/不明",
      "measurements": "測定値や数値データ",
      "notes": "特記事項"
    }}
  ],
  "issues_found": [
    {{
      "issue_id": "連番",
      "issue_description": "指摘事項の具体的内容", 
      "location": "発生場所・設備名",
      "severity": "緊急/重要/軽微",
      "category": "安全性/機能性/美観/法規制",
      "impact": "影響範囲や程度",
      "recommended_action": "推奨される対応策",
      "estimated_urgency": "対応期限の目安",
      "priority_score": "1-5の優先度スコア"
    }}
  ],
  "urgent_items": [
    {{
      "item": "緊急対応が必要な具体的内容",
      "risk_assessment": "リスクレベルと根拠", 
      "immediate_action": "即座に必要な応急措置",
      "deadline": "対応期限",
      "safety_concern": "安全面での懸念事項"
    }}
  ],
  "overall_condition": {{
    "general_status": "全体的な設備状況の評価",
    "trend_analysis": "前回からの変化や傾向",
    "maintenance_recommendations": "今後の保守提案",
    "compliance_status": "法規制への適合状況"
  }}
}}

**重要な分析ポイント:**
1. 数値データ（電流値、圧力、温度等）の基準値との比較
2. 安全性に関わる事項の優先度付け
3. 法定点検項目での不適合事項
4. 設備の劣化状況と交換時期の判断
5. 緊急性の高い修繕項目の特定

テキストが不完全な場合は「要確認」と記載し、推定できる範囲で分析してください。
必ずJSONフォーマットで回答し、日本語を使用してください。
"""
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.1,  # 一貫性を重視
            messages=[{
                "role": "user", 
                "content": analysis_prompt
            }]
        )
        
        # レスポンス内容を取得
        response_text = response.content[0].text
        
        # JSON部分を抽出（```json ... ``` で囲まれている場合）
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end]
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end]
        
        # JSONをパース
        try:
            analysis_result = json.loads(response_text.strip())
            return analysis_result
        except json.JSONDecodeError as e:
            st.error(f"JSON解析エラー: {e}")
            st.write("AIレスポンス:", response_text[:500] + "...")
            return {"error": "JSON解析失敗", "raw_response": response_text}
            
    except Exception as e:
        st.error(f"Anthropic API呼び出しエラー: {e}")
        return None

def process_csv_data(csv_file):
    """CSVファイルを処理して統計データを生成"""
    try:
        df = pd.read_csv(csv_file)
        
        # 基本統計
        total_operations = len(df)
        completed_operations = len(df[df['ステータス'] == '実施済'])
        
        # 指摘事項の分析
        operations_with_issues = df[df['対象ファイル'].str.contains('指摘有', na=False)]
        operations_no_issues = df[df['対象ファイル'].str.contains('指摘無', na=False)]
        
        # 担当会社
        companies = df['担当会社'].dropna().unique()
        
        # 緊急事項の検出（CSVメモ欄から）
        urgent_keywords = ['照明タイマー', 'シャッター', '鍵破損', '水漏れ', '異音', '破損', '不良']
        urgent_items = []
        
        for idx, row in operations_with_issues.iterrows():
            memo = str(row.get('メモ', ''))
            filename = str(row.get('対象ファイル', ''))
            
            for keyword in urgent_keywords:
                if keyword in memo or keyword in filename:
                    priority = "緊急" if keyword in ['照明タイマー', '鍵破損', '水漏れ'] else "短期"
                    urgent_items.append({
                        'priority': priority,
                        'operation': row['業務名'],
                        'company': row['担当会社'],
                        'issue': memo,
                        'date': row['日付'],
                        'source': 'CSV'
                    })
                    break
        
        stats = {
            "total_operations": total_operations,
            "completed_operations": completed_operations,
            "completion_rate": round(completed_operations / total_operations * 100, 1) if total_operations > 0 else 0,
            "companies_count": len(companies),
            "operations_with_issues": len(operations_with_issues),
            "operations_no_issues": len(operations_no_issues),
            "issue_rate": round(len(operations_with_issues) / total_operations * 100, 1) if total_operations > 0 else 0,
            "good_rate": round(len(operations_no_issues) / total_operations * 100, 1) if total_operations > 0 else 0
        }
        
        return {
            "stats": stats,
            "operations": df,
            "operations_with_issues": operations_with_issues,
            "operations_no_issues": operations_no_issues,
            "urgent_items": urgent_items,
            "companies": companies
        }
        
    except Exception as e:
        st.error(f"CSVファイルの処理中にエラーが発生しました: {e}")
        return None

def process_multiple_pdfs(client, pdf_files):
    """複数のPDFファイルをAI分析"""
    all_analysis = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_file in enumerate(pdf_files):
        status_text.text(f"分析中: {pdf_file.name} ({i+1}/{len(pdf_files)})")
        
        # PDFテキスト抽出
        pdf_text = extract_text_from_pdf(pdf_file)
        
        if pdf_text and len(pdf_text.strip()) > 50:
            # AI分析実行
            with st.spinner(f"AI分析中: {pdf_file.name}"):
                analysis = analyze_pdf_with_anthropic(client, pdf_text, pdf_file.name)
                
                if analysis and 'error' not in analysis:
                    analysis["filename"] = pdf_file.name
                    analysis["text_length"] = len(pdf_text)
                    all_analysis.append(analysis)
                else:
                    st.warning(f"{pdf_file.name} の分析でエラーが発生しました")
        else:
            st.warning(f"{pdf_file.name} からテキストを十分に抽出できませんでした")
        
        # プログレスバー更新
        progress_bar.progress((i + 1) / len(pdf_files))
    
    status_text.text("AI分析完了!")
    return all_analysis

def generate_enhanced_html_report(data, report_month, property_name, client_name, pdf_analysis_results=None):
    """PDF分析結果を統合したHTMLレポートを生成"""
    
    # 基本HTML構造は前回と同じだが、PDF分析結果を統合
    html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_month} {property_name} 月次業務報告レポート (AI Enhanced)</title>
    <style>
        /* 既存のCSS + 新しいPDF分析用スタイル */
        body {{
            font-family: 'Yu Gothic', 'Meiryo', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            font-size: 12px;
        }}
        .ai-analysis {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .severity-high {{ color: #dc3545; font-weight: bold; }}
        .severity-medium {{ color: #fd7e14; font-weight: bold; }}
        .severity-low {{ color: #6c757d; }}
        /* 他のCSSは前回と同じ */
        .header {{
            background-color: #495057;
            color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .section {{
            margin-bottom: 30px;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .section-header {{
            background-color: #6c757d;
            color: white;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_month} {property_name} 月次業務報告レポート</h1>
        <div class="ai-analysis">
            🤖 AI分析機能により詳細な指摘事項分析を実施
        </div>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <h3>総業務数</h3>
            <div class="number">{data['stats']['total_operations']}件</div>
        </div>
        <div class="card">
            <h3>AI分析PDF</h3>
            <div class="number">{len(pdf_analysis_results) if pdf_analysis_results else 0}件</div>
        </div>
        <div class="card">
            <h3>指摘事項</h3>
            <div class="number">{data['stats']['operations_with_issues']}件</div>
        </div>
        <div class="card">
            <h3>緊急対応</h3>
            <div class="number">{len(data.get('urgent_items', []))}件</div>
        </div>
    </div>
"""
    
    # PDF分析結果を追加
    if pdf_analysis_results:
        html_content += """
    <div class="section">
        <div class="section-header">🤖 AI分析による詳細指摘事項</div>
        <div style="padding: 20px;">
"""
        
        for analysis in pdf_analysis_results:
            if 'issues_found' in analysis and analysis['issues_found']:
                html_content += f"""
            <h4>📄 {analysis.get('filename', '不明ファイル')}</h4>
            <table class="inspection-table">
                <thead>
                    <tr>
                        <th>項目</th>
                        <th>場所</th>
                        <th>重要度</th>
                        <th>内容</th>
                        <th>推奨対応</th>
                    </tr>
                </thead>
                <tbody>
"""
                
                for issue in analysis['issues_found']:
                    severity_class = f"severity-{issue.get('severity', 'low').lower()}"
                    html_content += f"""
                    <tr>
                        <td>{issue.get('issue_id', '')}</td>
                        <td>{issue.get('location', '')}</td>
                        <td class="{severity_class}">{issue.get('severity', '')}</td>
                        <td>{issue.get('issue_description', '')}</td>
                        <td>{issue.get('recommended_action', '')}</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    return html_content

# メインアプリケーション
def main():
    st.title("月次業務報告レポート自動生成システム (AI Enhanced)")
    st.markdown("*Powered by Anthropic Claude API*")
    
    # サイドバーでの設定
    with st.sidebar:
        st.header("設定")
        
        # API KEY入力
        api_key_input = st.text_input(
            "Anthropic API KEY", 
            type="password",
            help="API KEYを入力するか、環境変数ANTHROPIC_API_KEYを設定してください"
        )
        
        if api_key_input:
            st.session_state.anthropic_api_key = api_key_input
        
        # レポート設定
        report_month = st.text_input("対象月", value="2025年7月度")
        property_name = st.text_input("物件名", value="グリーンオーク茅場町")
        client_name = st.text_input("発注者", value="双日ライフワン株式会社")
    
    # API設定チェック
    claude_client = check_anthropic_setup()
    
    if claude_client:
        st.success("Anthropic Claude API 接続確認")
        
        # ファイルアップロードセクション
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("業務データ (CSV)")
            csv_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])
        
        with col2:
            st.subheader("点検報告書 (PDF)")
            pdf_files = st.file_uploader(
                "PDFファイルをアップロード", 
                type=['pdf'],
                accept_multiple_files=True
            )
        
        # データ処理とレポート生成
        if csv_file is not None:
            st.markdown("---")
            
            # CSV処理
            with st.spinner("CSV業務データを処理中..."):
                processed_data = process_csv_data(csv_file)
            
            if processed_data:
                # PDF処理
                pdf_analysis_results = []
                if pdf_files:
                    st.subheader("AI PDF分析結果")
                    pdf_analysis_results = process_multiple_pdfs(claude_client, pdf_files)
                    
                    if pdf_analysis_results:
                        # 分析結果表示
                        for analysis in pdf_analysis_results:
                            with st.expander(f"📄 {analysis.get('filename', '不明ファイル')} - AI分析結果"):
                                
                                # 点検概要
                                if 'inspection_summary' in analysis:
                                    st.write("**点検概要:**")
                                    summary_df = pd.DataFrame([analysis['inspection_summary']])
                                    st.dataframe(summary_df, use_container_width=True)
                                
                                # 指摘事項
                                if 'issues_found' in analysis and analysis['issues_found']:
                                    st.write("**AI検出指摘事項:**")
                                    issues_df = pd.DataFrame(analysis['issues_found'])
                                    st.dataframe(issues_df, use_container_width=True)
                                
                                # 緊急事項
                                if 'urgent_items' in analysis and analysis['urgent_items']:
                                    st.write("**🚨 緊急対応事項:**")
                                    for urgent in analysis['urgent_items']:
                                        st.error(f"**{urgent.get('item', '')}**")
                                        st.write(f"リスク: {urgent.get('risk_assessment', '')}")
                                        st.write(f"対応: {urgent.get('immediate_action', '')}")
                
                # 統合レポート生成
                st.markdown("---")
                st.subheader("AI強化レポート生成")
                
                if st.button("🤖 AI分析レポートを生成", type="primary", use_container_width=True):
                    with st.spinner("AI強化HTMLレポートを生成中..."):
                        html_report = generate_enhanced_html_report(
                            processed_data, 
                            report_month, 
                            property_name, 
                            client_name,
                            pdf_analysis_results
                        )
                    
                    st.success("AI強化レポートが生成されました!")
                    
                    # ダウンロード
                    filename = f"{report_month}_{property_name}_AI強化月次レポート.html"
                    b64 = base64.b64encode(html_report.encode('utf-8')).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">📄 AI強化レポートをダウンロード</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # プレビュー
                    st.subheader("レポートプレビュー")
                    st.components.v1.html(html_report, height=800, scrolling=True)
        else:
            st.info("CSVファイルをアップロードしてください")

if __name__ == "__main__":
    main()
