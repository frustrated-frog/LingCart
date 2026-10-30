import pytest

from src.grounding import number_is_grounded


@pytest.mark.parametrize(
    "value,source",
    [
        (1, "一个键盘"),
        (2, "两个耳机"),
        (3, "三款商品"),
        (12, "十二件商品"),
        (20, "二十个候选"),
        (102, "一百零二元"),
        (400, "预算四百元"),
        (3.5, "重量三点五克"),
        (-2, "温度负二度"),
        (2026, "二〇二六年"),
    ],
)
def test_chinese_cardinal_numbers_are_grounded_by_value(value, source):
    assert number_is_grounded(value, [source])


def test_single_chinese_digit_inside_an_ordinary_word_is_not_a_number():
    assert not number_is_grounded(1, ["保持一致"])


def test_different_chinese_quantity_does_not_ground_the_model_value():
    assert not number_is_grounded(2, ["一个键盘"])
