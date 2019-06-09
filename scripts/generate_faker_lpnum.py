# coding=utf-8
from __future__ import unicode_literals

from faker import Faker
from faker.providers import BaseProvider

import string

localized = True
class InsurProvider(BaseProvider):
    license_plate_provinces = (
        "京","沪","浙","苏","粤","鲁","晋","冀",
        "豫","川","渝","辽","吉","黑","皖","鄂",
        "津","贵","云","桂","琼","青","新","藏",
        "蒙","宁","甘","陕","闽","赣","湘"
        )

    license_plate_last = ("挂","学","警","港","澳","使","领")

    license_plate_num = (
        "A","B","C","D","E","F","G","H",
        "J","K","L","M","N","P","Q","R",
        "S","T","U","V","W","X","Y","Z",
        "1","2","3","4","5","6","7","8","9","0"
        )

    def license_plate(self):
        """ 传统车牌 """
        plate = "{0}{1}{2}".format(
            self.random_element(self.license_plate_provinces),
            self.random_uppercase_letter(),
            "".join(self.random_choices(elements=self.license_plate_num, length=5))
        )
        return plate

    def special_license_plate(self):
        """ 特种车牌 """
        plate = "{0}{1}{2}{3}".format(
            self.random_element(self.license_plate_provinces),
            self.random_uppercase_letter(),
            "".join(self.random_choices(elements=self.license_plate_num, length=4)),
            self.random_element(self.license_plate_last)
        )
        return plate

    def custom_license_plate(self, prov, org, last=None):
        """
        prov: 省简称
        org: 发牌机关简称字母
        last: 特种车汉字字符
        """
        if last is None:
            plate = "{0}{1}{2}".format(prov, org, "".join(self.random_choices(elements=self.license_plate_num, length=5)))
        else:
            plate = "{0}{1}{2}{3}".format(prov, org, "".join(self.random_choices(elements=self.license_plate_num, length=4)),last)

        return plate

    def new_energy_license_plate(self, car_model=1):
        """ 
        新能源车牌 
        car_model: 车型，0-小型车，1-大型车
        """
        plate = ""
        if car_model == 0:
            # 小型车
            plate = "{0}{1}{2}{3}{4}".format(self.random_element(self.license_plate_provinces), self.random_uppercase_letter(),
                self.random_element(elements=("D", "F")), self.random_element(elements=self.license_plate_num), self.random_int(1000, 9999))
        else:
            # 大型车
            plate = "{0}{1}{2}{3}".format(self.random_element(self.license_plate_provinces), self.random_uppercase_letter(),
                self.random_int(10000, 99999), self.random_element(elements=("D", "F")))

        return plate

    def test_print(self):
        print(self.new_energy_license_plate())

if __name__ == "__main__":
    k = Faker()
    p = InsurProvider(k)
    
    # 随机生成普通车牌
    print(p.license_plate())

    # 随机生成特种车牌
    print(p.special_license_plate())

    # 自定义普通车牌
    print(p.custom_license_plate("湘", "A"))

    # 自定义特种车牌
    print(p.custom_license_plate("湘", "A", "挂"))

    # 随机生成新能源小型车车牌
    print(p.new_energy_license_plate(0))

    # 随机生成新能源大型车车牌
    print(p.new_energy_license_plate(1))
