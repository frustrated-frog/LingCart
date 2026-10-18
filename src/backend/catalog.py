"""Reproducible fictional products for the explicitly labeled demo catalog."""

from datetime import datetime, time, timedelta

from src.backend.db import Product, ShipmentEvent


CATEGORIES = ("耳机", "键盘", "扩展坞")
# Optional language metadata belongs to the catalog adapter, not the intent
# router. Categories without aliases remain discoverable through their live DB
# value and the semantic capability path.
CATEGORY_ALIASES = {
    "耳机": ("headphones", "earbuds"),
    "键盘": ("keyboard",),
    "扩展坞": ("usb-c hub", "usb c hub"),
}
SEARCH_ALIASES = {
    "noise cancelling": "降噪",
    "noise-cancelling": "降噪",
}
_AUDIO = [
    ("轻听", 199, "入耳式", "无", 24, "轻量通勤"),
    ("静行", 399, "入耳式", "主动降噪", 36, "地铁通勤"),
    ("云声", 499, "头戴式", "主动降噪", 40, "长时间办公"),
    ("畅跑", 299, "开放式", "无", 18, "运动佩戴"),
    ("听见", 599, "头戴式", "主动降噪", 50, "旅行降噪"),
    ("声桥", 249, "入耳式", "通话降噪", 28, "视频会议"),
    ("轻羽", 349, "开放式", "无", 30, "舒适通勤"),
    ("静界", 799, "头戴式", "主动降噪", 60, "长途旅行"),
    ("随听", 129, "入耳式", "无", 20, "日常听歌"),
    ("声场", 899, "头戴式", "主动降噪", 45, "音乐欣赏"),
]
_KEYBOARDS = [
    ("轻敲", 199, "有线", 87, "红轴", "日常办公"),
    ("静键", 329, "三模", 84, "静音轴", "低噪办公"),
    ("方格", 269, "双模", 98, "茶轴", "数字录入"),
    ("随行", 249, "三模", 68, "红轴", "紧凑桌面"),
    ("整列", 399, "三模", 108, "静音轴", "财务办公"),
    ("云键", 459, "三模", 75, "红轴", "多设备切换"),
    ("薄页", 299, "双模", 84, "矮轴", "轻薄便携"),
    ("击点", 549, "三模", 87, "磁轴", "游戏操作"),
    ("码序", 369, "有线", 87, "静音轴", "编程办公"),
    ("全域", 699, "三模", 98, "茶轴", "桌面定制"),
]
_HUBS = [
    ("连点", 99, 4, "无", 60, "日常外设"),
    ("视桥", 149, 6, "4K 30Hz", 100, "连接显示器"),
    ("多连", 229, 8, "4K 60Hz", 100, "桌面扩展"),
    ("轻接", 129, 5, "4K 30Hz", 65, "便携办公"),
    ("双屏", 399, 10, "双 HDMI 4K 60Hz", 100, "多屏办公"),
    ("网桥", 199, 7, "4K 30Hz", 100, "有线网络"),
    ("速连", 299, 8, "4K 60Hz", 100, "高速传输"),
    ("桌桥", 499, 12, "双 HDMI 4K 60Hz", 100, "固定工作台"),
    ("读卡", 89, 4, "无", 0, "摄影读卡"),
    ("全接", 699, 13, "双 HDMI 4K 60Hz", 140, "多设备办公"),
]


def product_fixtures() -> list[dict]:
    products = []
    for index, (name, price, form, noise, battery, use) in enumerate(_AUDIO):
        products.append(dict(product_id=f"P{1101 + index}", name=f"{name}耳机", category="耳机", price=price,
                             summary=f"{form} · {noise} · {use}",
                             specs={"佩戴方式": form, "降噪": noise, "综合续航": f"{battery} 小时", "适用场景": use}, stock=8 + index))
    for index, (name, price, connection, keys, switch, use) in enumerate(_KEYBOARDS):
        products.append(dict(product_id=f"P{1201 + index}", name=f"{name}机械键盘", category="键盘", price=price,
                             summary=f"{connection}连接 · {keys} 键 · {use}",
                             specs={"连接方式": connection, "键数": keys, "轴体": switch, "适用场景": use}, stock=12 + index))
    for index, (name, price, ports, video, power, use) in enumerate(_HUBS):
        products.append(dict(product_id=f"P{1301 + index}", name=f"{name} USB-C 扩展坞", category="扩展坞", price=price,
                             summary=f"{ports} 接口 · {video} · {use}",
                             specs={"接口数量": ports, "视频输出": video, "PD 供电": f"{power}W" if power else "不支持", "适用场景": use}, stock=6 + index))
    # Two distinct editions per model: useful comparison tradeoffs, not copies.
    editions = []
    for product in products:
        edition = {**product, "product_id": product["product_id"] + "S", "name": product["name"] + " 便携版",
                   "price": round(product["price"] * .85, 2), "specs": {**product["specs"], "版本": "便携版"},
                   "summary": product["summary"] + " · 便携收纳", "stock": product["stock"] // 2}
        if product["category"] == "耳机":
            hours = int(product["specs"]["综合续航"].split()[0])
            edition["specs"]["综合续航"] = f"{max(12, hours - 8)} 小时"
        elif product["category"] == "键盘":
            edition["specs"]["键数"] = min(68, product["specs"]["键数"])
            edition["summary"] = f"{edition['specs']['连接方式']}连接 · {edition['specs']['键数']} 键 · 便携收纳"
        else:
            edition["specs"]["接口数量"] = max(3, product["specs"]["接口数量"] - 2)
            edition["summary"] = f"{edition['specs']['接口数量']} 接口 · {edition['specs']['视频输出']} · 便携收纳"
        editions.append(edition)
    return products + editions


def seed_catalog(session) -> None:
    """Add missing demo products; never reset edited prices or stock."""
    for data in product_fixtures():
        if session.get(Product, data["product_id"]) is None:
            session.add(Product(**data))


def seed_shipments(session, orders) -> None:
    for order_id, _, _, status, order_date, _ in orders:
        descriptions = [(0, "订单已创建")]
        if status in {"shipped", "delivered"}:
            descriptions += [(1, "仓库已发货"), (2, "包裹运输中")]
        if status == "delivered":
            descriptions += [(4, "派送中"), (5, "包裹已签收")]
        elif status == "cancelled":
            descriptions += [(1, "订单已取消")]
        for index, (day, description) in enumerate(descriptions):
            event_id = f"{order_id}:{index}"
            if session.get(ShipmentEvent, event_id) is None:
                session.add(ShipmentEvent(event_id=event_id, order_id=order_id,
                                          occurred_at=datetime.combine(order_date + timedelta(days=day), time(10, 30)),
                                          description=description))
