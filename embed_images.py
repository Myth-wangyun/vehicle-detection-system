#!/usr/bin/env python3
"""将图片转换为Base64并嵌入HTML"""

import base64
import re

def image_to_base64(img_path):
    with open(img_path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

def main():
    # 读取HTML
    with open('docs/index.html', 'r') as f:
        html = f.read()
    
    # 图片列表
    images = {
        'training_curves.png': 'training_curves_b64',
        'experiment_results.png': 'experiment_results_b64',
        'model_weights.png': 'model_weights_b64',
        'validation_results.png': 'validation_results_b64',
    }
    
    # 转换并替换
    for img_name, var_name in images.items():
        try:
            b64 = image_to_base64(f'docs/{img_name}')
            html = html.replace(f'src="{img_name}"', f'src="{b64}"')
            print(f"Converted: {img_name} ({len(b64)} chars)")
        except FileNotFoundError:
            print(f"Not found: docs/{img_name}")
    
    # 保存
    with open('docs/index.html', 'w') as f:
        f.write(html)
    
    print("\nDone! Images embedded as Base64.")

if __name__ == '__main__':
    main()
