# ==================== 隐私过滤器 ====================
class PrivacyFilter:
    """隐私过滤器，处理窗口标题中的敏感信息"""
    
    def __init__(self, privacy_level="medium"):
        """
        privacy_level: low/medium/high
        - low: 保留最多信息
        - medium: 平衡隐私与效用（推荐）
        - high: 最大程度隐私保护
        """
        self.privacy_level = privacy_level
        self.browser_keywords = ["chrome", "firefox", "edge", "safari", "浏览器"]
        
    def filter_window_title(self, process_name: str, window_title: str) -> str:
        """过滤窗口标题中的隐私信息"""
        process_lower = process_name.lower()
        
        # 浏览器特殊处理（按你的"后门"设计）
        if any(browser in process_lower for browser in self.browser_keywords):
            return self._filter_browser_title(window_title)
        
        # 聊天软件处理
        elif any(app in process_lower for app in ["qq", "wechat", "微信", "telegram"]):
            return self._filter_chat_title(window_title)
        
        # 文档处理软件
        elif any(app in process_lower for app in ["word", "wps", "libreoffice", "记事本"]):
            return self._filter_document_title(window_title)
        
        # 代码编辑器
        elif any(app in process_lower for app in ["vscode", "pycharm", "idea", "sublime"]):
            return self._filter_code_editor_title(window_title)
        
        # 默认处理
        else:
            return self._filter_general_title(window_title)
    
    def _filter_browser_title(self, title: str) -> str:
        """浏览器标题过滤 - 保留站点信息"""
        if self.privacy_level == "high":
            return "网页浏览"
        elif self.privacy_level == "medium":
            # 尝试提取站点名
            for sep in [" - ", " – ", " | "]:
                if sep in title:
                    parts = title.split(sep)
                    if len(parts) >= 2:
                        site_part = parts[-2]  # 通常是站点名
                        # 简单清理
                        site_part = site_part.replace("https://", "").replace("http://", "")
                        site_part = site_part.split("/")[0].split("?")[0]
                        return f"浏览_{site_part[:20]}"
            return "网页浏览"
        else:  # low
            return title[:50]  # 仅截断，保留大部分信息
    
    def _filter_chat_title(self, title: str) -> str:
        """聊天软件标题过滤"""
        if self.privacy_level == "high":
            return "聊天窗口"
        elif self.privacy_level == "medium":
            # 移除具体联系人，保留类型
            if " - " in title:
                return "聊天窗口"
            return title[:30]
        else:  # low
            return title[:40]
    
    def _filter_document_title(self, title: str) -> str:
        """文档标题过滤"""
        if self.privacy_level == "high":
            return "文档编辑"
        elif self.privacy_level == "medium":
            # 提取文件类型
            if "." in title:
                ext = title.split(".")[-1].lower()
                if ext in ["doc", "docx", "pdf", "txt"]:
                    return f"{ext}文档"
            return "文档编辑"
        else:  # low
            return title[:40]
    
    def _filter_code_editor_title(self, title: str) -> str:
        """代码编辑器标题过滤"""
        if self.privacy_level == "high":
            return "代码编辑"
        elif self.privacy_level == "medium":
            # 提取项目或文件类型
            if " - " in title:
                main_part = title.split(" - ")[0]
                if "." in main_part:
                    ext = main_part.split(".")[-1].lower()
                    if ext in ["py", "js", "java", "cpp"]:
                        return f"{ext}代码"
                return "代码项目"
            return "代码编辑"
        else:  # low
            return title[:50]
    
    def _filter_general_title(self, title: str) -> str:
        """通用标题过滤"""
        if self.privacy_level == "high":
            return "应用程序"
        elif self.privacy_level == "medium":
            # 取第一个分隔符前的部分
            for sep in [" - ", " – ", " | ", " : "]:
                if sep in title:
                    return title.split(sep)[0][:30]
            return title[:30]
        else:  # low
            return title[:50]