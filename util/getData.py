# coding:utf-8
import numpy as np


def pre_processing():
    f = open(".\kddcup.data_10_percent_corrected")
    lines = f.readlines()
    data = list()
    protocol = ['tcp', 'icmp', 'udp']
    service = ['discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier',
               'link',
               'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames',
               'csnet_ns',
               'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs',
               'daytime',
               'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp',
               'ssh',
               'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher',
               'whois',
               'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec']
    dst_host_srv_rerror_rate = ['ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap',
                                'buffer_overflow', 'rootkit',
                                'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back',
                                'land', 'warezmaster',
                                'teardrop', 'warezclient']
    flag = ['OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1']

    for line in lines:
        info = line.split(",")
        # 这是将字符串value替换成数字value
        info[1] = protocol.index(info[1])
        info[2] = service.index(info[2])
        info[3] = flag.index(info[3])
        info[-1] = info[-1][:-2]
        info[-1] = dst_host_srv_rerror_rate.index(info[-1])
        # info = [int(float(x)) for x in info]

        data.append(info)
    f.close()

    a = np.array(data)
    np.save("a.npy", a)  # 保存为.npy格式

# glass数据集预处理以及生成npy
def glass_pre_processing():
    f = open("../dataSet/glass.data")
    lines = f.readlines()
    data = list()

    for line in lines:
        info = line.split(",")
        # 这是将字符串value替换成数字value
        info[-1] = info[-1][:-1]
        data.append(info)
    f.close()

    a = np.array(data)
    a = a[:, 1:]
    np.save("../dataSet/glass.npy", a)  # 保存为.npy格式


def get_data(path):
    # a = np.load("./networkData_number.npy")
    a = np.load(path)
    # a = a.tolist()
    return a


# 数据的处理  生成测试集和训练集
def data_split(X, y, rate=0.9):
    splitIndex = int(X.shape[0] * rate)
    train_X, train_y = X[:splitIndex], y[:splitIndex]
    test_X, test_y = X[splitIndex:], y[splitIndex:]

    return train_X, train_y, test_X, test_y


'''
 {'tcp', 'icmp', 'udp'}, 
 {'discard', 'IRC', 'rje', 'nnsp', 'klogin', 'netbios_ssn', 'iso_tsap', 'ldap', 'mtp', 'finger', 'courier', 'link', 'echo', 'time', 'remote_job', 'telnet', 'eco_i', 'tim_i', 'ecr_i', 'other', 'Z39_50', 'hostnames', 'csnet_ns', 'kshell', 'uucp_path', 'domain', 'nntp', 'uucp', 'bgp', 'pop_3', 'urp_i', 'auth', 'urh_i', 'efs', 'daytime', 'sunrpc', 'pm_dump', 'http', 'shell', 'http_443', 'systat', 'name', 'red_i', 'ntp_u', 'ftp_data', 'ftp', 'ssh', 'domain_u', 'netbios_ns', 'ctf', 'netbios_dgm', 'sql_net', 'printer', 'netstat', 'tftp_u', 'gopher', 'whois', 'imap4', 'login', 'supdup', 'smtp', 'pop_2', 'vmnet', 'private', 'X11', 'exec'}, 
 {'OTH', 'S0', 'S2', 'REJ', 'RSTR', 'SF', 'RSTOS0', 'S3', 'SH', 'RSTO', 'S1'}
 {'ipsweep', 'multihop', 'satan', 'ftp_write', 'guess_passwd', 'phf', 'spy', 'imap', 'buffer_overflow', 'rootkit', 'perl', 'normal', 'nmap', 'loadmodule', 'smurf', 'neptune', 'pod', 'portsweep', 'back', 'land', 'warezmaster', 'teardrop', 'warezclient'}
'''
if __name__ == '__main__':
    glass_pre_processing()
    # pre_processing()
    # data = get_data()
    # data = np.array(data)
    # t = list()
    # t.append(set(data[:, -1]))

    # pre_processing()
    # print(get_data())
