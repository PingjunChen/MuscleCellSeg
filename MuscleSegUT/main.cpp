/********************************************************************
    Copyright:  BICI2
    Created:    **:**:**** **:**
    Filename:   main.cpp
    Author:     Pingjun Chen

    Purpose:    unittest of UCM
*********************************************************************/

#include "gtest/gtest.h"

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int iRet = RUN_ALL_TESTS();

    return 0;
}
