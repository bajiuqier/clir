import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.header import Header
import logging

class SendEmail:
    def __init__(self, subject, content):
        # 设置日志记录
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # 从环境变量中获取 SMTP 服务器凭据
        self.mail_host = os.environ.get('MAIL_HOST', 'smtp.qq.com')
        self.mail_user = os.environ.get('MAIL_USER', 'yangher@foxmail.com')
        self.mail_pass = os.environ.get('MAIL_PASS', 'qcedvfianjxgbdda')

        # 如果缺少任何必需的环境变量,则引发异常
        if not self.mail_pass:
            raise ValueError('Missing required environment variable: MAIL_PASS')

        self.sender = self.mail_user
        self.receivers = ['bajiuqier@foxmail.com']
        self.subject = subject
        self.content = content

    def send_mail(self):
        try:
            # 创建安全的 SMTP 连接
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.mail_host, 465, context=context) as smtp:
                # 登录 SMTP 服务器
                smtp.login(self.mail_user, self.mail_pass)

                # 构建电子邮件
                msg = MIMEText(self.content, 'plain', 'utf-8')
                msg['From'] = Header(f'yangher <{self.sender}>')
                msg['To'] = Header(','.join(self.receivers), 'utf-8')
                msg['Subject'] = Header(self.subject, 'utf-8')

                # 发送电子邮件
                smtp.sendmail(self.sender, self.receivers, msg.as_string())
                logging.info('Email sent successfully')

        except Exception as e:
            logging.error(f'Error sending email: {e}')

if __name__ == '__main__':

    # 发送邮件提醒
    subject = '任务完成通知'
    content = '您的任务已经完成，请查收。'
    email = SendEmail(subject, content)
    email.send_mail()