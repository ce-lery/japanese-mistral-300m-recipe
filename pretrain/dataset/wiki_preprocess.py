import os


def main():
    # wiki.txtをよみこむ
    with open('wiki.txt', 'r') as f:
        lines = f.readlines()
        # 半角スペース、全角スペース、タブをすべて半角スペースに置換
        lines = [line.replace('\u3000', ' ').strip() for line in lines]
        lines = [line.replace('\t', ' ').strip() for line in lines]
        # 半角スペースが複数ある場合は一つに置換
        lines = [' '.join(line.split()) + '\n' for line in lines if line != '']
        # 半角スペースのみの行は削除
        lines = [line for line in lines if line != ' \n']
        # 1行の文字数が10文字以下の行は削除（juman++の処理に引っかかるのを防ぐため）
        lines = [line for line in lines if len(line) >= 20]

    with open('wiki_preprocess.txt', 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    main()
