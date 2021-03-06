#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# The zen of Python, by Tim Peters
# Beautiful is better than ugly.
# 优美胜于丑陋

# Explicit is better than implicit.
# 明了胜于晦涩

# Simple is better than complex.
# 简洁胜于复杂

# Complex is better than complicated.
# 复杂胜于凌乱

# Flat is better than nested.
# 扁平胜于嵌套

# Sparse is better than dense.
# 间隔胜于紧凑

# Readability counts.
# 可读性很重要

# Special cases aren't special enough to break the rules.
# Although practicality beats purity.
# 即使假借特例的实用性之名，也不可违背这些规则

# Errors should never pass silently.
# Unless explicitly silenced.
# 不要包容所有错误，除非你确定需要这么做

# In the face of ambiguity, refuse the temptation to guess.
# 当存在多种可能，不要尝试猜测

# There should be one-- and preferably only one --obvious way to do it.
# 而是尽量找一种，最好唯一一种明显的解决方案

# Although that way may not be obvious at first unless you're Dutch.

# 虽然这并不容易，因为你不是Python之父
# Now is better than never.
# Although never is often better than *right* now.
# 做也许好过不做，但不假思索就动手还不如不做

# If the implementation is hard to explain, it's a bad idea.
# If the implementation is easy to explain, it may be a good idea.
# 如果你无法向人描述你的方案，那肯定不是一个好方案；反之亦然

# Namespaces are one honking great idea -- let's do more of those!
# 命名空间是一种绝妙的理念，我们应当多加利用

# 导入mysql驱动
# pip3 install mysql-connector
import mysql.connector

conn = mysql.connector.connect(user='root', password='000000', database='python_db')
cursor = conn.cursor()
# 创建表
# cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
# 插入一行记录 占位符是%s
# cursor.execute('insert into user (id, name) values (%s, %s)', ['2', 'Bruce'])
cursor.rowcount

# 提交事务
conn.commit()
cursor.close()

# query data
cursor = conn.cursor()
cursor.execute('select * from user where id = %s', ('1',))
values = cursor.fetchall()
print(values)
# close cursor and conn
cursor.close()
conn.close

# test config
