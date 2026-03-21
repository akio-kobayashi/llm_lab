# Development Notes

## ノートブック更新日の自動反映
各 notebook の先頭に表示する `最終更新` は、ファイルごとの最終 commit 日から自動生成できます。

```bash
./scripts/install_git_hooks.sh
```

これを一度実行すると、以後 `git commit` の前に `scripts/update_notebook_dates.py` が自動実行され、更新後の notebook も自動で stage されます。
