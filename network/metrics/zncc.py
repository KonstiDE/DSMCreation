def getAverage(img, u, v, n):
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += img[u + i][v + j]
    return float(s) / (2 * n + 1) ** 2


def getStandardDeviation(img, u, v, n):
    s = 0
    avg = getAverage(img, u, v, n)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += (img[u + i][v + j] - avg) ** 2
    return (s ** 0.5) / (2 * n + 1)


def zncc(img1, img2, u1, v1, u2, v2, n):
    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
    stdDeviation2 = getStandardDeviation(img2, u2, v2, n)

    avg1 = getAverage(img1, u1, v1, n)
    avg2 = getAverage(img2, u2, v2, n)

    s = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += (img1[u1 + i][v1 + j] - avg1) * (img2[u2 + i][v2 + j] - avg2)
    return float(s) / ((2 * n + 1) ** 2 * stdDeviation1 * stdDeviation2)


if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B2 = [[1, 2, 3], [4, 5, 6], [7, 8, 7]]
    print(zncc(A, B1, 1, 1, 1, 1, 1))
    print(zncc(A, B2, 1, 1, 1, 1, 1))
